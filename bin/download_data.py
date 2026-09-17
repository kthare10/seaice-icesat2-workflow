#!/usr/bin/env python3

"""
Download ICESat-2 and Sentinel-2 data products for the sea ice workflow.

Unified CLI that fetches ATL03, ATL07, ATL10 granules from NASA Earthdata
and Sentinel-2 imagery from Microsoft Planetary Computer.

Paper defaults: Ross Sea region, November 2019, 8 reference ground tracks.

Usage:
    # Download everything with paper defaults
    python bin/download_data.py --all --output-dir ./data/

    # ICESat-2 only, specific products
    python bin/download_data.py --products atl03,atl07 --output-dir ./data/

    # Dry run — list granules without downloading
    python bin/download_data.py --all --dry-run --output-dir ./data/

    # Custom region and dates
    python bin/download_data.py --products atl03 --bbox -180,-78,-150,-60 \
        --start-date 2020-01-01 --end-date 2020-01-31 --output-dir ./data/

    # Filter to specific RGTs
    python bin/download_data.py --products atl03 --rgt 0578,0594,0731 --output-dir ./data/
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIONS = {
    "ross_sea": (-180, -78, -150, -60),
    "weddell_sea": (-60, -78, 0, -60),
    "beaufort_sea": (-160, 68, -120, 80),
    "arctic_ocean": (-180, 65, 180, 90),
    "southern_ocean": (-180, -78, 180, -60),
}

# RGTs from the paper (Ross Sea, November 2019)
# RGTs of the eight IS2/S2 coincident pairs in the paper's Table I (Nov 3, 4, 13, 16, 17, 20,
# 23, 26 2019). 0578/0594/0731/0838/0883/0929 are confirmed by the author's ATL03 filenames
# (e.g. ATL03_20191123180255_08830510); 0777 and 0792 are derived from the Table I acquisition
# times (15.24 RGTs/day).
PAPER_RGTS = ["0578", "0594", "0731", "0777", "0792", "0838", "0883", "0929"]

ICESAT2_PRODUCTS = {
    "atl03": "ATL03",
    "atl07": "ATL07",
    "atl10": "ATL10",
}

ALL_PRODUCTS = list(ICESAT2_PRODUCTS.keys()) + ["sentinel2"]

# ---------------------------------------------------------------------------
# Earthdata authentication
# ---------------------------------------------------------------------------


def authenticate_earthdata():
    """Authenticate with NASA Earthdata.

    Returns:
        (use_token, session) — if use_token is True, session has a bearer
        header set; otherwise earthaccess.login() was called and callers
        should use earthaccess.download().
    """
    import requests

    token = os.environ.get("EARTHDATA_TOKEN")
    if token:
        logger.info("Using pre-generated EARTHDATA_TOKEN (bearer token)")
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        return True, session

    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        import earthaccess

        logger.info("Using EARTHDATA_USERNAME/PASSWORD login")
        earthaccess.login(strategy="environment")
        return False, None

    raise RuntimeError(
        "NASA Earthdata credentials not set. Provide either:\n"
        "  - EARTHDATA_TOKEN (pre-generated bearer token), or\n"
        "  - EARTHDATA_USERNAME and EARTHDATA_PASSWORD\n"
        "Register at https://urs.earthdata.nasa.gov/"
    )


# ---------------------------------------------------------------------------
# ICESat-2 search & download
# ---------------------------------------------------------------------------


def search_icesat2(product, bbox, start_date, end_date, rgt_filter=None):
    """Search for ICESat-2 granules via CMR.

    Args:
        product: Short name key (e.g. "atl03")
        bbox: (min_lon, min_lat, max_lon, max_lat)
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        rgt_filter: Optional list of 4-digit RGT strings to keep

    Returns:
        List of earthaccess granule results.
    """
    import earthaccess

    short_name = ICESAT2_PRODUCTS[product]
    logger.info(f"Searching {short_name} granules  bbox={bbox}  {start_date} → {end_date}")

    results = earthaccess.search_data(
        short_name=short_name,
        bounding_box=bbox,
        temporal=(start_date, end_date),
    )

    if rgt_filter:
        # ATL filenames encode RGT as characters 22-25:
        # ATL03_20191103184432_05780510_006_02.h5
        #                      ^^^^
        rgt_set = set(rgt_filter)
        filtered = []
        for granule in results:
            links = granule.data_links(access="external")
            if not links:
                continue
            fname = Path(links[0]).name
            # Extract RGT from filename using regex
            m = re.search(r"ATL\d{2}_\d{14}_(\d{4})", fname)
            if m and m.group(1) in rgt_set:
                filtered.append(granule)
        logger.info(
            f"RGT filter {sorted(rgt_set)}: {len(results)} → {len(filtered)} granules"
        )
        results = filtered

    logger.info(f"Found {len(results)} {short_name} granules")
    return results


def download_icesat2(product, granules, output_dir, use_token, session, dry_run=False):
    """Download ICESat-2 granules to output_dir/<product>/.

    Individual granule HDF5 files are saved (no merging).

    Returns:
        List of downloaded file Paths.
    """
    short_name = ICESAT2_PRODUCTS[product]
    dest = Path(output_dir) / product
    dest.mkdir(parents=True, exist_ok=True)

    if dry_run:
        logger.info(f"[DRY RUN] Would download {len(granules)} {short_name} granules to {dest}")
        for i, g in enumerate(granules):
            links = g.data_links(access="external")
            name = Path(links[0]).name if links else f"granule_{i}"
            logger.info(f"  [{i+1}/{len(granules)}] {name}")
        return []

    downloaded = []

    if use_token:
        import requests
        from requests.exceptions import ChunkedEncodingError, ConnectionError, SSLError

        max_retries = 5
        base_delay = 5

        for i, granule in enumerate(granules):
            urls = granule.data_links(access="external")
            if not urls:
                logger.warning(f"No download URL for {short_name} granule {i}")
                continue
            url = urls[0]
            filename = Path(url).name
            out_path = dest / filename

            # Resume: skip files that already exist with nonzero size
            if out_path.exists() and out_path.stat().st_size > 0:
                logger.info(f"  [{i+1}/{len(granules)}] Skipping {filename} (already exists)")
                downloaded.append(out_path)
                continue

            logger.info(f"  [{i+1}/{len(granules)}] Downloading {filename}...")

            for attempt in range(1, max_retries + 1):
                try:
                    resp = session.get(url, stream=True, timeout=300)
                    if resp.status_code != 200:
                        logger.warning(f"  HTTP {resp.status_code} for {url}")
                        break
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    size_mb = out_path.stat().st_size / 1e6
                    logger.info(f"  [{i+1}/{len(granules)}] Saved {filename} ({size_mb:.1f} MB)")
                    downloaded.append(out_path)
                    break
                except (SSLError, ConnectionError, ChunkedEncodingError) as e:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"  Attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    session.close()
                    session = __import__("requests").Session()
                    token = os.environ.get("EARTHDATA_TOKEN", "")
                    session.headers.update({"Authorization": f"Bearer {token}"})
                    if attempt < max_retries:
                        time.sleep(delay)
                    else:
                        logger.error(f"  All {max_retries} attempts failed for {filename}")
    else:
        import earthaccess

        files = earthaccess.download(granules, local_path=str(dest))
        downloaded = [Path(f) for f in files]

    logger.info(f"Downloaded {len(downloaded)} {short_name} files to {dest}")
    return downloaded


# ---------------------------------------------------------------------------
# Sentinel-2 download
# ---------------------------------------------------------------------------


def download_sentinel2(bbox, start_date, end_date, output_dir, max_cloud_cover=30,
                       max_scenes=20, dry_run=False):
    """Download Sentinel-2 L2A scenes from Planetary Computer.

    Returns:
        List of downloaded file Paths.
    """
    try:
        import planetary_computer
        import pystac_client
        import rasterio
    except ImportError as exc:
        logger.warning(
            f"Sentinel-2 download skipped — missing dependency: {exc}. "
            f"Install with: pip install pystac-client planetary-computer rasterio"
        )
        return []

    dest = Path(output_dir) / "sentinel2"
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Searching Sentinel-2 L2A  bbox={bbox}  {start_date} → {end_date}  cloud<{max_cloud_cover}%")

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
    )

    items = list(search.items())
    logger.info(f"Found {len(items)} Sentinel-2 scenes")

    if not items:
        logger.warning("No Sentinel-2 scenes found for the specified criteria")
        return []

    # Sort by cloud cover (prefer clearest)
    items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
    items = items[:max_scenes]

    if dry_run:
        logger.info(f"[DRY RUN] Would download {len(items)} Sentinel-2 scenes to {dest}")
        for i, item in enumerate(items):
            cc = item.properties.get("eo:cloud_cover", -1)
            logger.info(f"  [{i+1}/{len(items)}] {item.id}  cloud={cc:.1f}%")
        return []

    bands = ["B02", "B03", "B04", "B08"]  # Blue, Green, Red, NIR
    import shutil
    import tarfile

    tmp_dir = dest / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    downloaded_scenes = []

    for i, item in enumerate(items):
        scene_id = item.id
        cc = item.properties.get("eo:cloud_cover", -1)
        logger.info(f"  [{i+1}/{len(items)}] {scene_id}  cloud={cc:.1f}%")

        scene_dir = tmp_dir / scene_id
        scene_dir.mkdir(exist_ok=True)

        for band_name in bands:
            if band_name not in item.assets:
                logger.warning(f"    Band {band_name} not found in {scene_id}")
                continue
            asset = item.assets[band_name]
            out_path = scene_dir / f"{band_name}.tif"
            try:
                with rasterio.open(asset.href) as src:
                    data = src.read()
                    profile = src.profile.copy()
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)
                logger.info(f"    Saved {band_name}")
            except Exception as e:
                logger.warning(f"    Failed {band_name}: {e}")
                continue

        downloaded_scenes.append(scene_dir)

    # Package into tar.gz
    tar_path = dest / "sentinel2_scenes.tar.gz"
    logger.info(f"Packaging {len(downloaded_scenes)} scenes into {tar_path}")
    with tarfile.open(tar_path, "w:gz") as tar:
        for scene_dir in downloaded_scenes:
            tar.add(scene_dir, arcname=scene_dir.name)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    logger.info(f"Sentinel-2 data saved to {tar_path}")
    return [tar_path]


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(results):
    """Print a formatted summary of downloaded data."""
    print(f"\n{'=' * 70}")
    print("DOWNLOAD SUMMARY")
    print(f"{'=' * 70}")
    total = 0
    for product, files in sorted(results.items()):
        count = len(files)
        total += count
        if count > 0:
            total_mb = sum(f.stat().st_size for f in files if f.exists()) / 1e6
            print(f"  {product.upper():12s}  {count:4d} files  ({total_mb:,.1f} MB)")
        else:
            print(f"  {product.upper():12s}  {count:4d} files")
    print(f"{'=' * 70}")
    print(f"  {'TOTAL':12s}  {total:4d} files")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Download ICESat-2 and Sentinel-2 data for the sea ice workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything (paper defaults: Ross Sea, Nov 2019)
  %(prog)s --all --output-dir ./data/

  # Download only ICESat-2 ATL03
  %(prog)s --products atl03 --output-dir ./data/

  # Dry run — list granules without downloading
  %(prog)s --all --dry-run --output-dir ./data/

  # Custom region and dates
  %(prog)s --products atl03 --bbox -180,-78,-150,-60 \\
      --start-date 2020-01-01 --end-date 2020-01-31 --output-dir ./data/

  # Filter to specific RGTs from the paper
  %(prog)s --products atl03 --rgt 0578,0594,0731 --output-dir ./data/
        """,
    )

    parser.add_argument(
        "--products", type=str, default="atl03",
        help="Comma-separated products: atl03, atl07, atl10, sentinel2 (default: atl03)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all products (atl03, atl07, atl10, sentinel2)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory (created if needed)",
    )
    parser.add_argument(
        "--region", type=str, default="ross_sea",
        help=f"Predefined region name (default: ross_sea). Options: {', '.join(REGIONS)}",
    )
    parser.add_argument(
        "--bbox", type=str, default=None,
        help="Custom bounding box: min_lon,min_lat,max_lon,max_lat (overrides --region)",
    )
    parser.add_argument(
        "--start-date", type=str, default="2019-11-01",
        help="Start date YYYY-MM-DD (default: 2019-11-01)",
    )
    parser.add_argument(
        "--end-date", type=str, default="2019-11-30",
        help="End date YYYY-MM-DD (default: 2019-11-30)",
    )
    parser.add_argument(
        "--rgt", type=str, default=None,
        help="Comma-separated RGT filter, e.g. 0578,0594 (ICESat-2 only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List granules/scenes without downloading",
    )
    parser.add_argument(
        "--max-granules", type=int, default=None,
        help="Max granules per ICESat-2 product (for testing)",
    )
    parser.add_argument(
        "--max-cloud-cover", type=float, default=30,
        help="Sentinel-2 max cloud cover %% (default: 30)",
    )
    parser.add_argument(
        "--max-scenes", type=int, default=20,
        help="Max Sentinel-2 scenes (default: 20)",
    )

    args = parser.parse_args()

    # Resolve products
    if args.all:
        products = ALL_PRODUCTS[:]
    else:
        products = [p.strip().lower() for p in args.products.split(",")]
        for p in products:
            if p not in ALL_PRODUCTS:
                parser.error(f"Unknown product '{p}'. Choose from: {', '.join(ALL_PRODUCTS)}")

    # Resolve bounding box
    if args.bbox:
        parts = [float(x) for x in args.bbox.split(",")]
        if len(parts) != 4:
            parser.error("--bbox must have 4 comma-separated values: min_lon,min_lat,max_lon,max_lat")
        bbox = tuple(parts)
    else:
        if args.region not in REGIONS:
            parser.error(f"Unknown region '{args.region}'. Options: {', '.join(REGIONS)}")
        bbox = REGIONS[args.region]

    # Resolve RGT filter
    rgt_filter = None
    if args.rgt:
        rgt_filter = [r.strip().zfill(4) for r in args.rgt.split(",")]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Products: {products}")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Date range: {args.start_date} → {args.end_date}")
    if rgt_filter:
        logger.info(f"RGT filter: {rgt_filter}")
    if args.dry_run:
        logger.info("DRY RUN — no files will be downloaded")

    # Authenticate for ICESat-2 products (only if needed)
    icesat2_products = [p for p in products if p in ICESAT2_PRODUCTS]
    use_token, session = False, None
    if icesat2_products and not args.dry_run:
        use_token, session = authenticate_earthdata()
    elif icesat2_products and args.dry_run:
        # CMR search works without auth; still need earthaccess for search
        try:
            import earthaccess  # noqa: F401
        except ImportError:
            logger.error("earthaccess is required for ICESat-2 search. pip install earthaccess")
            sys.exit(1)

    results = {}

    # Download ICESat-2 products
    for product in icesat2_products:
        try:
            granules = search_icesat2(product, bbox, args.start_date, args.end_date, rgt_filter)
            if args.max_granules and len(granules) > args.max_granules:
                logger.info(f"Limiting {product} to {args.max_granules} granules")
                granules = granules[: args.max_granules]
            files = download_icesat2(
                product, granules, output_dir, use_token, session, dry_run=args.dry_run,
            )
            results[product] = files
        except Exception as e:
            logger.error(f"Failed to download {product}: {e}")
            import traceback
            traceback.print_exc()
            results[product] = []

    # Download Sentinel-2
    if "sentinel2" in products:
        try:
            files = download_sentinel2(
                bbox, args.start_date, args.end_date, output_dir,
                max_cloud_cover=args.max_cloud_cover,
                max_scenes=args.max_scenes,
                dry_run=args.dry_run,
            )
            results["sentinel2"] = files
        except Exception as e:
            logger.error(f"Failed to download sentinel2: {e}")
            import traceback
            traceback.print_exc()
            results["sentinel2"] = []

    if not args.dry_run:
        print_summary(results)

    logger.info("Done.")


if __name__ == "__main__":
    main()
