# T2 Radiomics Extraction

Source notebook:

- `ipynb/Pre_operative(T2_radiomics).ipynb`

Role:

- downstream analysis notebook for radiomics extraction
- consumes generated and real T2-side outputs

Observed inputs:

- base path points into:
  - `datasets/T1W2T2W/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations`
- generated image example:
  - `fake_T2_epoch_200.nrrd`
- real image example:
  - `real_T2.nrrd`

Observed radiomics stack:

- `pyradiomics`
- `SimpleITK`
- `featureextractor.RadiomicsFeatureExtractor`

Observed flow:

1. load case path
2. load reconstructed real or fake T2 image
3. load mask image
4. run `RadiomicsFeatureExtractor`
5. export per-case feature CSV

Observed output examples:

- `real_T2.csv`
- `fake_T2.csv`

Notes:

- this notebook is downstream of image translation, not part of the first
  translation implementation milestone
- it is still important because the broader project depends on generated T2
  images being usable for radiomics extraction
- the current repository now includes a scriptable PNG-to-NRRD reconstruction
  bridge before this notebook stage:
  - `scripts/postprocess/reconstruct_t2_volumes.py`
- `contour.nrrd` is still a separate artifact and is not yet reconstructed by
  that script
