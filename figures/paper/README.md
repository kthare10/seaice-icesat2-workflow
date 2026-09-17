# Figures reproduced from the paper

The images in this directory are panels extracted from:

> Iqrah, J. M., Koo, Y., Wang, W., Xie, H. & Prasad, S. K. (2025). *Scalable Higher Resolution Polar
> Sea Ice Classification and Freeboard Calculation from ICESat-2 ATL03 Data.* IEEE International
> Parallel and Distributed Processing Symposium Workshops (IPDPSW).
> <https://arxiv.org/abs/2502.02700>

They are included solely so that [`../../PAPER_COMPARISON.md`](../../PAPER_COMPARISON.md) can show
this workflow's results beside the published ones.

**Licence.** The paper is distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). These panels carry that same
licence and are reproduced here with attribution for non-commercial comparison. They are **not**
covered by whatever licence applies to the rest of this repository.

| File | Paper figure |
|---|---|
| `fig04_confusion_matrix.png` | Fig. 4, sea-ice classification confusion matrix |
| `fig06a_classification_atl03_nov04_gt2r.png` | Fig. 6a, ATL03 classification, track 20191104195311_05940510 gt2r |
| `fig06b_classification_atl07_nov04_gt2r.png` | Fig. 6b, ATL07 classification (Koo method), same track |
| `fig07a_classification_atl03_nov26_gt2r.png` | Fig. 7a, ATL03 classification, track 20191126182014_09290510 gt2r |
| `fig08a_sea_surface_methods_nov04_gt2r.png` | Fig. 8a, local sea surface by four methods |
| `fig10a_freeboard_atl03_nov04_gt2r.png` | Fig. 10a, ATL03 freeboard along track |
| `fig10d_point_density_nov04_gt2r.png` | Fig. 10d, ATL03 vs ATL07 point density |

To regenerate them from the PDF:

```bash
pdfimages -png -p 2502.02700v1.pdf pf
# then identify the panels; the mapping above records which extracted image is which
```

The matching panels from this workflow are in [`../run/`](../run/), rendered by
[`../../bin/comparison_figures.py`](../../bin/comparison_figures.py).
