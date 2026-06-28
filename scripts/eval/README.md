# Evaluation Scripts

This directory contains executable evaluation utilities for generated image
artifacts.

Current target:

- paired `real_B` / `fake_B` PNG evaluation for the `T1 <-> T2` baseline
- paired `real_T2.nrrd` / `fake_T2_epoch_200.nrrd` evaluation for existing case directories

Current behavior:

- scan a directory for paired image files
- match pairs by filename stem
- choose the comparison pair from CycleGAN direction
- compute image-level metrics
- write per-pair CSV output
- write aggregate summary CSV output

Metric policy:

- `RMSE`: direct array-space root mean squared error
- `Mutual Information`: binned grayscale mutual information using
  `scikit-learn`
- `PSNR`: `scikit-image`
- `SSIM`: `scikit-image`

Example command:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/eval/t1_t2_image_metrics.py `
  --image-root outputs/generated/t1_t2_cyclegan_2d/generation_test/images `
  --output-dir outputs/metrics/t1_t2_cyclegan_2d/generation_test `
  --direction AtoB
```

Legacy-style example:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/eval/t1_t2_image_metrics.py `
  --image-root results/TCGA-02-0006/T1W2T2W_batch_1_cyclegan/test_200/images `
  --output-dir outputs/metrics/t1_t2_cyclegan_2d/legacy_case_TCGA-02-0006 `
  --direction AtoB
```

Existing case-directory NRRD example:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/eval/t1_t2_nrrd_metrics.py `
  --real-volume data_set/BraTS2017TestingData/TCGA-02-0003/real_T2.nrrd `
  --fake-volume data_set/BraTS2017TestingData/TCGA-02-0003/fake_T2_epoch_200.nrrd `
  --output-dir outputs/metrics/t1_t2_cyclegan_2d/TCGA-02-0003_nrrd
```

Batch NRRD example:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/eval/t1_t2_nrrd_metrics_batch.py `
  --cases-root data_set/BraTS2017TestingData `
  --output-dir outputs/metrics/t1_t2_cyclegan_2d/all_cases_nrrd
```

Ranking example:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/eval/t1_t2_case_rankings.py `
  --case-summary-csv outputs/metrics/t1_t2_cyclegan_2d/all_cases_nrrd/case_summary_metrics.csv `
  --output-dir outputs/metrics/t1_t2_cyclegan_2d/all_cases_nrrd/rankings `
  --top-k 5
```

Direction mapping:

- `AtoB` evaluates `real_B` against `fake_B`
- `BtoA` evaluates `real_A` against `fake_A`

Output files:

- `per_pair_metrics.csv`
- `summary_metrics.csv`
- `per_slice_metrics.csv`
- `case_summary_metrics.csv`
- `overall_summary_metrics.csv`
- `top_rmse_cases.csv`
- `bottom_mi_cases.csv`
- `bottom_ssim_cases.csv`
- `case_rankings.md`

Important note:

- the historical notebooks confirm the metric names, but not the exact original
  code cell
- this script is therefore a reproducible standard-library reimplementation,
  not a byte-for-byte notebook recovery
- in this workspace, setting `UV_CACHE_DIR` inside the repository is the safest
  way to avoid permission issues with the default global uv cache
