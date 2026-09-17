#!/usr/bin/env python3
"""
Pegasus workflow generator for sea ice classification and freeboard calculation
from ICESat-2 ATL03 data (Iqrah et al., IPDPSW 2025).

Containers
----------
The workflow builds its own Apptainer images. Two build jobs run `apptainer build`
on the definition files in Apptainer/, and the resulting .sif images are ordinary
workflow files: Pegasus stages each one to exactly the jobs that declare it as an
input. No Docker, no registry, and nothing to prepare by hand before planning.

Nothing from this repository is copied into an image; the definitions have no
%files section. Each compute job instead stages three things - the
bin/seaice_run.sh wrapper, its bin/<stage>.py script and the .sif - and the
wrapper bind-mounts the job directory into the container. Editing a stage script
therefore never invalidates an image.

Pass --prebuilt-containers DIR to reuse images you have already built and skip the
build jobs.

Pipeline stages:
0. Build containers (CPU, GPU) - run on the submit host, staged to workers
1. (Optional) Extract ATL07/ATL10 CSVs from raw HDF5 (fan-out per granule)
2. (Optional) Auto-label correction - trim edge open-water chunks (skipped for *_done.csv)
3. Preprocess ATL03 - relative sea surface, interpolation, h_diff (fan-out per input file)
4. Prepare LSTM data - sliding-window feature vectors, merged per unique track (fan-in)
5. Train LSTM - 3-class classifier (GPU); model, norm params, metrics, test predictions
6. Inference LSTM - batch prediction (GPU)
7. Compute freeboard - NASA sea surface equation + smoothing (fan-out per unique track)
8. Validate freeboard - ATL03 vs ATL07/ATL10 comparison (fan-out per unique track)
9. Generate paper figures - tables and plots (fan-in)

A "track" is a unique granule+beam, e.g. 20191104195311_05940510_gt1r. Several input
files (Sentinel-2 tiles) may belong to one track; they are merged in stage 4.

Usage:
    python workflow_generator.py --input-dir ./data/IS2_Corrected_data/ -o workflow.yml
    python workflow_generator.py --input-dir ./data/labeled/ --preset paper -o workflow.yml
    python workflow_generator.py --input-dir ./data/ --prebuilt-containers ./containers -o workflow.yml
"""

import argparse
import glob
import logging
import os
import re
import sys
from pathlib import Path

from Pegasus.api import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRACK_RE = re.compile(r"ATL03_(\d{14})_(\d{8})_.*?(gt\d[lr])", re.IGNORECASE)
BEAM_RE = re.compile(r"(gt\d[lr])", re.IGNORECASE)

# Container images the workflow builds, and the files each build job needs staged.
# Each image is built from its definition alone - the definitions have no %files
# section, so nothing from this repository is copied inside. Stage scripts are
# bind-mounted at run time instead, which is why editing workflow code never
# invalidates an image.
CONTAINERS = {
    "cpu": {"sif": "seaice_cpu.sif", "definition": "Apptainer/seaice_cpu.def"},
    "gpu": {"sif": "seaice_gpu.sif", "definition": "Apptainer/seaice_gpu.def"},
}

# --horovod swaps the GPU image for one that also carries OpenMPI and Horovod.
# It is a separate definition because the Horovod build compiles from source and
# adds roughly 10-20 minutes to a cold build.
HOROVOD_CONTAINER = {"sif": "seaice_gpu_horovod.sif",
                     "definition": "Apptainer/seaice_gpu_horovod.def"}

# stage name -> (script, container, memory, gpus)
STAGES = {
    "extract_atl07":      ("extract_atl07.py", "cpu", "4 GB", 0),
    "extract_atl10":      ("extract_atl10.py", "cpu", "4 GB", 0),
    "autolabel_correct":  ("autolabel_correct.py", "cpu", "4 GB", 0),
    "preprocess_atl03":   ("preprocess_atl03.py", "cpu", "8 GB", 0),
    "prepare_lstm_data":  ("prepare_lstm_data.py", "cpu", "12 GB", 0),
    "train_lstm":         ("train_lstm.py", "gpu", "12 GB", 1),
    "inference_lstm":     ("inference_lstm.py", "gpu", "12 GB", 1),
    "compute_freeboard":  ("compute_freeboard.py", "cpu", "8 GB", 0),
    "validate_freeboard": ("validate_freeboard.py", "cpu", "4 GB", 0),
    "paper_figures":      ("paper_figures.py", "cpu", "8 GB", 0),
}


def track_id_of(filename):
    """Unique track id <datetime>_<rgtcycle>_<beam>; mirrors bin/prepare_lstm_data.py."""
    base = os.path.basename(filename)
    m = TRACK_RE.search(base)
    if m:
        return f"{m.group(1)}_{m.group(2)}_{m.group(3).lower()}"
    m = BEAM_RE.search(base)
    if m:
        return m.group(1).lower()
    return os.path.splitext(base)[0]


class SeaIceClassificationWorkflow:
    """Sea ice classification and freeboard Pegasus workflow generator."""

    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_dir = None
    shared_scratch_dir = None
    local_storage_dir = None
    wf_name = "seaice-classification-freeboard"

    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_dir = str(Path(__file__).parent.resolve())
        self.shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        self.local_storage_dir = os.path.join(self.wf_dir, "output")
        # LFN -> File for the wrapper/stage scripts and container inputs
        self.script_files = {}
        self.sif_files = {}

    def write(self):
        """Write all catalogs and workflow to files."""
        if self.sc is not None:
            self.sc.write()
        self.props.write()
        self.rc.write()
        self.tc.write()
        self.wf.write(file=self.dagfile)

    def create_pegasus_properties(self):
        """Create Pegasus properties configuration."""
        self.props = Properties()
        self.props["pegasus.data.configuration"] = "condorio"
        self.props["pegasus.transfer.bypass.input.staging"] = "true"
        self.props["pegasus.transfer.threads"] = "16"
        # Use the submit host's worker package instead of downloading one per job.
        self.props["pegasus.transfer.worker.package"] = "true"

    def create_sites_catalog(self, exec_site_name="condorpool"):
        """Create site catalog. The container build jobs are pinned to 'local'."""
        logger.info("Creating site catalog for: %s", exec_site_name)
        self.sc = SiteCatalog()

        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, self.shared_scratch_dir).add_file_servers(
                FileServer("file://" + self.shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, self.local_storage_dir).add_file_servers(
                FileServer("file://" + self.local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = (
            Site(exec_site_name)
            .add_condor_profile(universe="vanilla")
            .add_pegasus_profile(style="condor")
            .add_pegasus_profile(data_configuration="condorio")
        )

        self.sc.add_sites(local, exec_site)

    def select_containers(self, horovod=False):
        """Swap in the Horovod GPU image when distributed training is requested."""
        self.containers = {k: dict(v) for k, v in CONTAINERS.items()}
        if horovod:
            self.containers["gpu"] = dict(HOROVOD_CONTAINER)
            logger.info("Horovod requested: GPU image is %s", HOROVOD_CONTAINER["definition"])
        return self.containers

    def create_transformation_catalog(self, n_gpus=1, horovod=False, build_containers=True):
        """Register the build wrapper and one stageable transformation per pipeline stage.

        Every stage shares bin/seaice_run.sh as its executable; the distinct
        transformation names keep the per-stage breakdown in pegasus-statistics and
        carry each stage's memory and GPU requests.
        """
        logger.info("Creating transformation catalog")
        self.tc = TransformationCatalog()

        runner_pfn = os.path.join(self.wf_dir, "bin/seaice_run.sh")

        if build_containers:
            # The build runs on the submit host, so these are submit-host paths.
            # A persistent cache means the base image layers are pulled once, not
            # once per workflow run; the tmpdir needs room for the unpacked image.
            cache_dir = os.environ.get("APPTAINER_CACHEDIR",
                                       os.path.expanduser("~/.apptainer/cache"))
            tmp_dir = os.environ.get("APPTAINER_TMPDIR",
                                     os.path.join(self.wf_dir, ".apptainer_tmp"))
            build_tr = Transformation(
                "build_container", site="local",
                pfn=os.path.join(self.wf_dir, "bin/build_container.sh"),
                is_stageable=True,
            ).add_pegasus_profile(memory="8 GB").add_env(
                APPTAINER_CACHEDIR=cache_dir,
                APPTAINER_TMPDIR=tmp_dir,
            )
            self.tc.add_transformations(build_tr)
            logger.info("Container build cache: %s (tmp: %s)", cache_dir, tmp_dir)

        for name, (_script, kind, memory, gpus) in STAGES.items():
            tr = Transformation(
                name, site="local", pfn=runner_pfn, is_stageable=True,
            ).add_pegasus_profile(memory=memory)
            if gpus:
                request = n_gpus if (horovod and name == "train_lstm") else gpus
                tr.add_condor_profile(request_gpus=request)
            self.tc.add_transformations(tr)

    def create_replica_catalog(self, input_files, atl07_files=None, atl10_files=None,
                               raw_atl07_files=None, raw_atl10_files=None,
                               build_containers=True, prebuilt_dir=None):
        """Register data inputs, the stage scripts, and either the container
        definitions (to build) or pre-built .sif images."""
        self.rc = ReplicaCatalog()

        def add(path):
            lfn = os.path.basename(path)
            self.rc.add_replica("local", lfn, "file://" + os.path.abspath(path))
            return File(lfn)

        for filelist in (input_files, atl07_files or [], atl10_files or [],
                         raw_atl07_files or [], raw_atl10_files or []):
            for filepath in filelist:
                add(filepath)
        logger.info("Replica catalog: %d data files", len(input_files))

        # Stage scripts travel with their jobs, so a code change needs no rebuild.
        for _name, (script, _kind, _mem, _gpus) in STAGES.items():
            script_path = os.path.join(self.wf_dir, "bin", script)
            if not os.path.isfile(script_path):
                raise SystemExit(f"Stage script not found: {script_path}")
            self.script_files[script] = add(script_path)

        if build_containers:
            self.container_build_inputs = {}
            for kind, spec in self.containers.items():
                def_path = os.path.join(self.wf_dir, spec["definition"])
                if not os.path.isfile(def_path):
                    raise SystemExit(f"Container definition not found: {def_path}")
                self.container_build_inputs[kind] = [add(def_path)]
                # Produced by the build job, so only a File handle is needed here.
                self.sif_files[kind] = File(spec["sif"])
            logger.info("Containers: built in-workflow from %s",
                        ", ".join(c["definition"] for c in self.containers.values()))
        else:
            for kind, spec in self.containers.items():
                sif_path = os.path.join(prebuilt_dir, spec["sif"])
                if not os.path.isfile(sif_path):
                    raise SystemExit(
                        f"--prebuilt-containers: {sif_path} not found. Build it with\n"
                        f"  apptainer build {sif_path} {spec['definition']}\n"
                        f"or drop the flag to let the workflow build it.")
                self.sif_files[kind] = add(sif_path)
            logger.info("Containers: pre-built, from %s", prebuilt_dir)

    # ------------------------------------------------------------------
    # Job helper
    # ------------------------------------------------------------------

    def stage_job(self, stage, job_id, args, node_label=None, np=None):
        """Build a Job that runs one stage script inside its staged container.

        Wires up the wrapper's own inputs (the .sif and the stage script), adds
        --nv for GPU stages, and --np N to launch under horovodrun. `args` are the
        stage script's arguments.
        """
        script, kind, _memory, gpus = STAGES[stage]
        sif = self.sif_files[kind]
        script_file = self.script_files[script]

        job = Job(stage, _id=job_id, node_label=node_label or job_id)
        wrapper_args = ["--container", sif, "--script", script_file]
        if gpus:
            wrapper_args.append("--nv")
        if np:
            wrapper_args += ["--np", str(np)]
        wrapper_args.append("--")
        job.add_args(*wrapper_args, *args)
        job.add_inputs(sif, script_file)
        return job

    # ------------------------------------------------------------------
    # DAG
    # ------------------------------------------------------------------

    def create_workflow(self, input_files, epochs=None, batch_size=32, preset="notebook",
                        units=None, dropout=None, lr=None, radius=5000,
                        weight_form="paper", smooth_window=10000, zscore=False,
                        horovod=False, n_gpus=1, skip_autolabel=False,
                        build_containers=True,
                        atl07_files=None, atl10_files=None,
                        raw_atl07_files=None, raw_atl10_files=None,
                        paper_dates=("20191104", "20191126")):
        """Create the workflow DAG."""
        logger.info("Creating workflow DAG")
        self.wf = Workflow(self.wf_name, infer_dependencies=True)

        # --- Stage 0: build the container images -------------------------------
        # Pinned to the submit host: unprivileged `apptainer build` needs
        # user-namespace support the compute nodes lack, and building here avoids
        # shipping a multi-GB image back from a worker before redistributing it.
        if build_containers:
            for kind, spec in self.containers.items():
                sif = self.sif_files[kind]
                def_file = self.container_build_inputs[kind][0]
                build_job = (
                    Job("build_container", _id=f"build_container_{kind}",
                        node_label=f"build_container_{kind}")
                    .add_args("--def", def_file, "--output", sif)
                    .add_inputs(*self.container_build_inputs[kind])
                    .add_outputs(sif, stage_out=False, register_replica=False)
                    .add_selector_profile(execution_site="local")
                )
                self.wf.add_jobs(build_job)

        # --- Stage 1: extract ATL07/ATL10 from raw HDF5 (optional) --------------
        atl07_input_files = []
        atl10_input_files = []
        for product, raw_files, sink in (("atl07", raw_atl07_files, atl07_input_files),
                                         ("atl10", raw_atl10_files, atl10_input_files)):
            for h5_path in raw_files or []:
                h5_lfn = os.path.basename(h5_path)
                stem = os.path.splitext(h5_lfn)[0]
                h5_file = File(h5_lfn)
                csv_file = File(stem + ".csv")
                job = self.stage_job(
                    f"extract_{product}", f"extract_{product}_{stem}",
                    ["--input", h5_file, "--output-dir", "."])
                job.add_inputs(h5_file)
                job.add_outputs(csv_file, stage_out=False, register_replica=False)
                self.wf.add_jobs(job)
                sink.append(csv_file)
        for apath in atl07_files or []:
            atl07_input_files.append(File(os.path.basename(apath)))
        for apath in atl10_files or []:
            atl10_input_files.append(File(os.path.basename(apath)))

        # --- Stages 2-3: per input file ----------------------------------------
        enriched_files = []
        track_ids = []
        for filepath in input_files:
            lfn = os.path.basename(filepath)
            input_file = File(lfn)
            base = os.path.splitext(lfn)[0]
            tid = track_id_of(lfn)
            if tid not in track_ids:
                track_ids.append(tid)

            if not skip_autolabel:
                corrected_file = File(base + "_corrected.csv")
                job = self.stage_job(
                    "autolabel_correct", f"autolabel_{base}",
                    ["--input", input_file, "--output", corrected_file])
                job.add_inputs(input_file)
                job.add_outputs(corrected_file, stage_out=False, register_replica=False)
                self.wf.add_jobs(job)
                preprocess_input = corrected_file
            else:
                preprocess_input = input_file

            enriched_file = File(base + "_enriched.csv")
            pre_args = ["--input", preprocess_input, "--output", enriched_file,
                        "--radius", str(int(radius))]
            if zscore:
                pre_args.append("--zscore")
            job = self.stage_job("preprocess_atl03", f"preprocess_{base}", pre_args)
            job.add_inputs(preprocess_input)
            job.add_outputs(enriched_file, stage_out=False, register_replica=False)
            self.wf.add_jobs(job)
            enriched_files.append(enriched_file)

        logger.info("Unique tracks (granule+beam): %d -> %s", len(track_ids), track_ids)

        # --- Stage 4: prepare LSTM data (fan-in; merges tiles of one track) -----
        prepared_csv = File("ATL03_cor_label_LSTM_ann_prepared.csv")
        prepare_job = self.stage_job(
            "prepare_lstm_data", "prepare_lstm_data",
            ["--input-dir", ".", "--output", prepared_csv, "--nearby", "2"])
        for ef in enriched_files:
            prepare_job.add_inputs(ef)
        prepare_job.add_outputs(prepared_csv, stage_out=True, register_replica=False)
        self.wf.add_jobs(prepare_job)

        # --- Stage 5: train LSTM (GPU) -----------------------------------------
        model_file = File("lstm_sea_ice_model.h5")
        norm_params = File("norm_params.json")
        metrics_file = File("training_metrics.json")
        test_pred_file = File("test_predictions.csv")

        train_args = [
            "--data", prepared_csv,
            "--model-output", model_file,
            "--norm-params-output", norm_params,
            "--metrics-output", metrics_file,
            "--test-predictions-output", test_pred_file,
            "--preset", preset,
            "--batch-size", str(batch_size),
        ]
        for flag, val in (("--epochs", epochs), ("--units", units),
                          ("--dropout", dropout), ("--lr", lr)):
            if val is not None:
                train_args += [flag, str(val)]
        if horovod:
            train_args.append("--horovod")

        train_job = self.stage_job("train_lstm", "train_lstm", train_args,
                                   np=n_gpus if horovod else None)
        train_job.add_inputs(prepared_csv)
        train_job.add_outputs(model_file, norm_params, metrics_file, test_pred_file,
                              stage_out=True, register_replica=False)
        self.wf.add_jobs(train_job)

        # --- Stage 6: inference LSTM (GPU) -------------------------------------
        predictions_csv = File("LSTM_predictions_all.csv")
        inference_job = self.stage_job(
            "inference_lstm", "inference_lstm",
            ["--model", model_file, "--data", prepared_csv,
             "--norm-params", norm_params, "--output", predictions_csv])
        inference_job.add_inputs(model_file, norm_params, prepared_csv)
        inference_job.add_outputs(predictions_csv, stage_out=True, register_replica=False)
        self.wf.add_jobs(inference_job)

        # --- Stage 7: compute freeboard (fan-out per unique track) --------------
        freeboard_files = []
        for tid in track_ids:
            fb_out = File(f"freeboard_{tid}.csv")
            job = self.stage_job(
                "compute_freeboard", f"freeboard_{tid}",
                ["--input", predictions_csv, "--output", fb_out,
                 "--radius", str(int(radius)), "--method", "nasa_sea_surface_eqn",
                 "--weight-form", weight_form, "--smooth-window", str(int(smooth_window)),
                 "--track", tid])
            job.add_inputs(predictions_csv)
            job.add_outputs(fb_out, stage_out=True, register_replica=False)
            self.wf.add_jobs(job)
            freeboard_files.append((tid, fb_out))

        # --- Stage 8: validate freeboard (fan-out per unique track) -------------
        for tid, fb_file in freeboard_files:
            report_file = File(f"report_{tid}.tar.gz")
            val_args = ["--atl03-freeboard", fb_file, "--output-dir", ".",
                        "--output-tar", report_file, "--track", tid]
            extra_inputs = []
            # Wire only the ATL07/ATL10 files matching this track's date (YYYYMMDD)
            date_match = re.match(r"(\d{8})", tid)
            track_date = date_match.group(1) if date_match else None
            for flag, pool in (("--atl07-dir", atl07_input_files),
                               ("--atl10-dir", atl10_input_files)):
                if pool:
                    matched = [f for f in pool if track_date and track_date in str(f)] or pool
                    val_args.extend([flag, "."])
                    extra_inputs.extend(matched)
            job = self.stage_job("validate_freeboard", f"validate_{tid}", val_args)
            job.add_inputs(fb_file, *extra_inputs)
            job.add_outputs(report_file, stage_out=True, register_replica=False)
            self.wf.add_jobs(job)

        # --- Stage 9: paper figures (fan-in) ------------------------------------
        figures_tar = File("paper_figures.tar.gz")
        fig_args = ["--all", "--predictions", predictions_csv,
                    "--test-predictions", test_pred_file,
                    "--output-dir", ".", "--output-tar", figures_tar]
        extra_inputs = [predictions_csv, test_pred_file]
        if freeboard_files:
            fig_args.append("--freeboard")
            for _, fb_file in freeboard_files:
                fig_args.append(fb_file)
                extra_inputs.append(fb_file)
        # Only the ATL07/ATL10 CSVs for the paper's two figure tracks (avoids OOM)
        for flag, pool in (("--atl07-csv", atl07_input_files),
                           ("--atl10-csv", atl10_input_files)):
            if pool:
                matched = [f for f in pool if any(d in str(f) for d in paper_dates)] or pool
                fig_args.append(flag)
                for f in matched:
                    fig_args.append(f)
                    extra_inputs.append(f)
        figures_job = self.stage_job("paper_figures", "paper_figures", fig_args)
        figures_job.add_inputs(*extra_inputs)
        figures_job.add_outputs(figures_tar, stage_out=True, register_replica=False)
        self.wf.add_jobs(figures_job)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pegasus workflow for sea ice classification and freeboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input-dir ./data/IS2_Corrected_data/ -o workflow.yml
  %(prog)s --input-dir ./data/IS2_Corrected_data/ --preset paper -o workflow.yml
  %(prog)s --input-dir ./data/ --prebuilt-containers ./containers -o workflow.yml
        """,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing labeled track CSVs")
    parser.add_argument("--pattern", default="*_labeled_10m_done.csv",
                        help="Glob pattern for input CSV files (default: *_labeled_10m_done.csv)")
    parser.add_argument("-s", "--skip-sites-catalog", action="store_true")
    parser.add_argument("-e", "--execution-site-name", default="condorpool",
                        help="Execution site name (default: condorpool)")
    parser.add_argument("-o", "--output", default="workflow.yml",
                        help="Output file (default: workflow.yml)")
    parser.add_argument("--preset", choices=["notebook", "paper"], default="notebook",
                        help="LSTM architecture preset (default: notebook = NB3 tuned model)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: preset value; notebook 50, paper 20)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--units", type=int, default=None, help="LSTM units (overrides preset)")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (overrides preset)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides preset)")
    parser.add_argument("--radius", type=float, default=5000,
                        help="Sliding window radius in metres (default: 5000)")
    parser.add_argument("--weight-form", choices=["paper", "notebook"], default="paper",
                        help="Lead weight formula for the NASA sea surface (default: paper)")
    parser.add_argument("--smooth-window", type=int, default=10000,
                        help="+/-rows for sea-surface smoothing (Notebook 6: 10000; 0 disables)")
    parser.add_argument("--zscore", action="store_true",
                        help="Enable the notebook's disabled z-score stage in preprocess_atl03")
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--horovod", action="store_true",
                        help="Data-parallel training: use the Horovod GPU image and launch "
                             "train_lstm under `horovodrun -np <n-gpus>` on one worker")
    parser.add_argument("--prebuilt-containers", metavar="DIR", default=None,
                        help="Reuse seaice_cpu.sif / seaice_gpu.sif from DIR and skip the "
                             "container build jobs")
    autolabel = parser.add_mutually_exclusive_group()
    autolabel.add_argument("--skip-autolabel", action="store_true",
                           help="Skip auto-label correction (default when inputs are *_done.csv)")
    autolabel.add_argument("--force-autolabel", action="store_true",
                           help="Run auto-label correction even for *_done.csv inputs")
    parser.add_argument("--max-tracks", type=int, default=None,
                        help="Limit to first N input files (for quick test runs)")
    parser.add_argument("--atl07-dir", default=None,
                        help="Directory containing ATL07 CSV files (ATL07*.csv) for validation")
    parser.add_argument("--atl10-dir", default=None,
                        help="Directory containing ATL10 CSV files (ATL10*.csv) for validation")
    parser.add_argument("--raw-atl07-dir", default=None,
                        help="Directory containing raw ATL07 HDF5 files (*.h5) to extract")
    parser.add_argument("--raw-atl10-dir", default=None,
                        help="Directory containing raw ATL10 HDF5 files (*.h5) to extract")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not input_files:
        logger.error("No input files found in %s matching %s", args.input_dir, args.pattern)
        sys.exit(1)

    if args.max_tracks is not None:
        input_files = input_files[:args.max_tracks]
        logger.info("--max-tracks %d: using %d input files", args.max_tracks, len(input_files))

    # Already corrected inputs (*_done.csv) must not be trimmed again.
    all_done = all(os.path.basename(f).endswith("_done.csv") for f in input_files)
    skip_autolabel = args.skip_autolabel or (all_done and not args.force_autolabel)
    if all_done and not args.skip_autolabel and not args.force_autolabel:
        logger.info("All inputs are *_done.csv (already corrected): skipping auto-label correction")

    build_containers = args.prebuilt_containers is None

    if args.horovod:
        logger.info("--horovod: training runs as `horovodrun -np %d` inside %s",
                    args.n_gpus, HOROVOD_CONTAINER["definition"])
        if args.n_gpus < 2:
            logger.warning("--horovod with --n-gpus %d trains on a single rank; pass --n-gpus 2 "
                           "or more for actual data parallelism.", args.n_gpus)
        logger.warning("All %d ranks must fit on ONE worker: HTCondor's vanilla universe gives the "
                       "job a single machine, so --n-gpus cannot exceed a worker's GPU count.",
                       args.n_gpus)

    def discover(d, pattern, what):
        if not d:
            return []
        files = sorted(glob.glob(os.path.join(d, pattern)))
        if not files:
            logger.warning("No %s files found in %s", what, d)
        return files

    atl07_files = discover(args.atl07_dir, "ATL07*.csv", "ATL07*.csv")
    atl10_files = discover(args.atl10_dir, "ATL10*.csv", "ATL10*.csv")
    raw_atl07_files = discover(args.raw_atl07_dir, "*.h5", "ATL07 *.h5")
    raw_atl10_files = discover(args.raw_atl10_dir, "*.h5", "ATL10 *.h5")

    logger.info("=" * 70)
    logger.info("SEA ICE CLASSIFICATION & FREEBOARD WORKFLOW GENERATOR")
    logger.info("=" * 70)
    logger.info("Input files: %d", len(input_files))
    logger.info("Preset: %s, Epochs: %s, Batch size: %d",
                args.preset, args.epochs or "preset", args.batch_size)
    logger.info("Radius: %.0f m, weights: %s, smooth window: %d, z-score: %s",
                args.radius, args.weight_form, args.smooth_window, args.zscore)
    logger.info("GPUs: %d, Horovod: %s", args.n_gpus, args.horovod)
    logger.info("Containers: %s", "built by the workflow" if build_containers
                else f"pre-built from {args.prebuilt_containers}")
    logger.info("Skip autolabel: %s", skip_autolabel)
    logger.info("ATL07 CSV: %d, ATL10 CSV: %d, raw ATL07: %d, raw ATL10: %d",
                len(atl07_files), len(atl10_files), len(raw_atl07_files), len(raw_atl10_files))
    logger.info("Execution site: %s", args.execution_site_name)
    logger.info("=" * 70)

    try:
        wf = SeaIceClassificationWorkflow(dagfile=args.output)
        wf.select_containers(horovod=args.horovod)
        if not args.skip_sites_catalog:
            wf.create_sites_catalog(args.execution_site_name)
        wf.create_pegasus_properties()
        wf.create_transformation_catalog(n_gpus=args.n_gpus, horovod=args.horovod,
                                         build_containers=build_containers)
        wf.create_replica_catalog(input_files, atl07_files=atl07_files, atl10_files=atl10_files,
                                  raw_atl07_files=raw_atl07_files, raw_atl10_files=raw_atl10_files,
                                  build_containers=build_containers,
                                  prebuilt_dir=args.prebuilt_containers)
        wf.create_workflow(
            input_files=input_files, epochs=args.epochs, batch_size=args.batch_size,
            preset=args.preset, units=args.units, dropout=args.dropout, lr=args.lr,
            radius=args.radius, weight_form=args.weight_form, smooth_window=args.smooth_window,
            zscore=args.zscore, horovod=args.horovod, n_gpus=args.n_gpus,
            skip_autolabel=skip_autolabel, build_containers=build_containers,
            atl07_files=atl07_files, atl10_files=atl10_files,
            raw_atl07_files=raw_atl07_files, raw_atl10_files=raw_atl10_files,
        )
        wf.write()

        logger.info("=" * 70)
        logger.info("WORKFLOW GENERATION COMPLETE")
        logger.info("  1. Review: %s", args.output)
        logger.info("  2. Submit: pegasus-plan --submit -s %s -o local %s",
                    args.execution_site_name, args.output)
        logger.info("  3. Monitor: pegasus-status <submit_dir>")
        logger.info("=" * 70)
    except Exception as e:
        logger.error("Failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
