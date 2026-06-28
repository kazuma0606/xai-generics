# Evaluation Notes

## Confirmed Legacy Image Metrics

The current workflow discussion confirms these image-generation evaluation
metrics:

- RMSE
- Mutual Information
- PSNR
- SSIM

These are the currently confirmed baseline metrics.

Implementation assumption confirmed by the user:

- the metric implementations were primarily taken from standard Python
  libraries rather than custom research code
- `scikit-learn` is the baseline assumption for reusable metric components
- `PSNR` and `SSIM` should be treated as likely `scikit-image` metrics in the
  future executable implementation, because those are the standard library
  entrypoints typically used for them

## Observed Legacy File Flow

The current notebook evidence supports the following file-level evaluation flow:

1. CycleGAN test outputs are stored under a path like:
   - `results/<case>/T1W2T2W_batch_1_cyclegan/test_200/images`
2. per-slice generated or reference images are read from PNG patterns such as:
   - `*_<slice>_real_B.png`
3. downstream volume-style files exist per case as:
   - `real_T2.nrrd`
   - `fake_T2_epoch_200.nrrd`
   - `contour.nrrd`
4. the radiomics notebook reads those `.nrrd` files as `(155, 240, 240)` arrays
5. PyRadiomics is executed with the contour mask and writes:
   - `CSV/real_T2.csv`
   - `CSV/fake_T2.csv`

Interpretation:

- image-generation evaluation likely happens before or alongside the `.nrrd`
  packaging stage
- radiomics evaluation clearly happens after per-case volume-style files already
  exist
- the exact RMSE / Mutual Information / PSNR / SSIM calculation cell is still
  not isolated from the notebooks and remains pending

Current audit result:

- a direct code-cell search across the local `.ipynb` files did not surface any
  explicit RMSE / Mutual Information / PSNR / SSIM implementation
- the current notebook evidence only confirms the surrounding file flow and the
  radiomics-side `.nrrd` consumption
- the exact metric implementation may have existed in another notebook, an
  untracked script, or a later manual evaluation step outside the checked-in
  notebooks
- for the first reproducible reimplementation, using standard Python library
  implementations is consistent with the user's recollection even if the
  original cell is not recovered

## Historical FID Attempt

There was also a historical attempt to evaluate generated images with FID, but
it was abandoned.

Reference discussed by the user:

- https://zenn.dev/fmuuly/articles/e53cbed26fa927

Practical interpretation:

- keep FID as a documented idea, not as a required first milestone
- if it is retried later, document exactly what feature extractor and image
  preprocessing are used

## Why Plain FID May Be Unsatisfying Here

Likely reasons this was not a good fit for the legacy workflow:

- MRI slices are not natural-image RGB data in the usual FID sense
- pseudo-RGB inputs can distort what the embedding network is really measuring
- 2D slice-level FID may not reflect patient-level structural fidelity
- downstream usefulness for radiomics/prognosis may diverge from FID

## Candidate Evaluation Improvements

These are candidate additions for later evaluation design, not fixed
requirements yet.

- patient-level aggregation instead of slice-only scoring
- modality-aware feature comparison instead of only natural-image embeddings
- radiomics feature correlation between real and generated outputs
- downstream task sensitivity:
  compare whether prognosis-related radiomics behavior is preserved
- paired perceptual metrics such as LPIPS if a medically defensible setup is
  defined
- ROI-aware evaluation using the available tumor masks

## Recommended Evaluation Layers

For this project, evaluation is likely stronger if it is layered.

Layer 1:

- RMSE
- Mutual Information
- PSNR
- SSIM

Layer 2:

- radiomics feature agreement between real and generated images

Layer 3:

- downstream prognosis stability through Lasso-Cox / Logrank analysis

This structure is closer to the real research goal than relying on one image
metric alone.
