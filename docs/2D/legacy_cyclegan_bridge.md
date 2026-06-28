# Legacy CycleGAN Bridge

## Purpose

This note defines the handoff point between the current repository and the
external legacy CycleGAN codebase that the notebooks used historically.

## Current Boundary

This repository currently provides:

- raw-data assumptions
- notebook-derived settings
- reproducible preprocessing from `.nii.gz` to paired 2D slices
- manifests for slice provenance

This repository does not currently provide:

- `train.py`
- `test.py`
- legacy dataset classes
- CycleGAN model code

That code is still expected to come from an external
`pytorch-CycleGAN-and-pix2pix` style checkout.

## Expected Dataroot Layout

The most direct bridge is the legacy flat layout:

```text
outputs/derived/t1_t2_cyclegan_legacy/
  trainA/
  trainB/
  testA/
  testB/
  train_manifest.csv
  generation_test_manifest.csv
```

This is now produced by `scripts/prepare/t1_t2_2d_prepare.py` when
`--flat-output` is used with:

- `--split-name train`
- `--split-name generation_test`

## Minimal Handoff Procedure

1. Prepare `trainA/trainB` from
   `data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations`.
2. Prepare `testA/testB` from `data_set/BraTS2017TestingData`.
3. Point the external CycleGAN `--dataroot` to
   `outputs/derived/t1_t2_cyclegan_legacy`.
4. Keep the first reproduction at the observed legacy settings:
   - `batch_size=1`
   - `load_size=286`
   - `crop_size=256`
   - `preprocess=resize_and_crop`
   - `n_epochs=100`
   - `n_epochs_decay=100`

## Example Preparation Commands

```powershell
uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations `
  --output-root outputs/derived/t1_t2_cyclegan_legacy `
  --split-name train `
  --write-png `
  --skip-empty `
  --flat-output
```

```powershell
uv run python scripts/prepare/t1_t2_2d_prepare.py `
  --input-root data_set/BraTS2017TestingData `
  --output-root outputs/derived/t1_t2_cyclegan_legacy `
  --split-name generation_test `
  --write-png `
  --skip-empty `
  --flat-output
```

## Known Blocker

The remaining blocker is not dataset preparation.

The remaining blocker is that the actual CycleGAN training/inference project is
not present in this repository, so an end-to-end train/test smoke run cannot be
executed here yet.
