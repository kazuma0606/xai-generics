# Prepare Scripts

This directory contains dataset-preparation utilities that replace notebook-only
data export steps.

Current target:

- `T1 <-> T2` 2D paired slice preparation

Current behavior:

- read canonical `.nii.gz` volumes
- extract paired axial slices
- preserve patient ID and slice index
- optionally write cached PNG exports
- always write a manifest for reproducibility

Runtime:

- use `uv run` from the repository root
- dependencies are defined in `pyproject.toml`

Example commands:

```powershell
uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations `
  --output-root outputs/derived/t1_t2_2d `
  --split-name train `
  --write-png `
  --skip-empty
```

```powershell
uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/BraTS2017TestingData `
  --output-root outputs/derived/t1_t2_2d `
  --split-name generation_test `
  --write-png `
  --skip-empty `
  --limit-cases 2
```

Output layout:

- `outputs/derived/t1_t2_2d/<split-name>/A`
- `outputs/derived/t1_t2_2d/<split-name>/B`
- `outputs/derived/t1_t2_2d/<split-name>/manifest.csv`

Legacy CycleGAN-compatible layout is also supported:

```powershell
uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations `
  --output-root outputs/derived/t1_t2_cyclegan_legacy `
  --split-name train `
  --write-png `
  --skip-empty `
  --flat-output

uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/BraTS2017TestingData `
  --output-root outputs/derived/t1_t2_cyclegan_legacy `
  --split-name generation_test `
  --write-png `
  --skip-empty `
  --flat-output
```

That produces:

- `outputs/derived/t1_t2_cyclegan_legacy/trainA`
- `outputs/derived/t1_t2_cyclegan_legacy/trainB`
- `outputs/derived/t1_t2_cyclegan_legacy/testA`
- `outputs/derived/t1_t2_cyclegan_legacy/testB`
- `outputs/derived/t1_t2_cyclegan_legacy/train_manifest.csv`
- `outputs/derived/t1_t2_cyclegan_legacy/generation_test_manifest.csv`

Smoke test result:

- `train_smoke`: manifest-only run completed on `1` training case
- `generation_test_smoke`: manifest-only run completed on `1` case
- `generation_test_smoke_png`: paired PNG export completed on `1` case
- `t1_t2_cyclegan_legacy_smoke`: paired PNG export completed on `1` case with
  direct `testA/testB` output
- observed first generation smoke-test case: `TCGA-02-0003`
- observed slice count for each current smoke-test case: `155`
