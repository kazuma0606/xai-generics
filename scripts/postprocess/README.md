# Postprocess Scripts

This directory contains downstream artifact conversion utilities.

Current target:

- reconstruct case-level T2 volumes from generated PNG slices

Current behavior:

- scan a case image directory for paired `real_B` / `fake_B` PNG files
- choose the reconstruction pair from CycleGAN direction
- sort slices by numeric slice index
- convert RGB-like PNGs into scalar grayscale slices
- stack slices into 3D arrays with shape `(slice, height, width)`
- write:
  - `real_T2.nrrd`
  - `fake_T2_epoch_200.nrrd`

Example command:

```powershell
$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'
uv run python scripts/postprocess/reconstruct_t2_volumes.py `
  --image-root results/TCGA-02-0006/T1W2T2W_batch_1_cyclegan/test_200/images `
  --output-dir outputs/radiomics/t1_t2_cyclegan_2d/TCGA-02-0006 `
  --direction AtoB
```

Important note:

- this is a reproducible reconstruction bridge for downstream radiomics
- `AtoB` expects `real_B` and `fake_B`
- `BtoA` expects `real_A` and `fake_A`
- it does not yet reconstruct `contour.nrrd`; the mask still needs to come
  from the raw case segmentation source
- in this workspace, setting `UV_CACHE_DIR` inside the repository is the safest
  way to avoid permission issues with the default global uv cache
