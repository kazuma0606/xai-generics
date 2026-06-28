# 2D Baseline Tasks

## Current Priority

Establish a clean, local, reproducible 2D baseline starting from raw
`data_set/` volumes and legacy notebook settings.

The first execution target is the plain `T1 <-> T2` baseline.
The first `test` target means generation inputs for the trained model.

## Tasks

- [x] Confirm the final path naming convention for 2D baseline assets under `docs/`, `scripts/`, and output directories.
- [x] Extract the exact slice-generation logic from `CycleGAN_batch1_RGB_T1_T2_.ipynb`.
- [x] Extract the legacy T1/T2 slice-generation and pairing logic from the notebook workflow.
- [x] Recreate the plain `T1 <-> T2` baseline without notebook-only manual steps.
- [x] Add a local paired T1/T2 dataset loader for 2D training.
- [x] Add a local CycleGAN-compatible model/checkpoint wrapper scaffold that preserves generator weights.
- [x] Add a local train CLI scaffold for the plain `T1 <-> T2` baseline.
- [x] Add a local inference CLI scaffold for the plain `T1 <-> T2` baseline.
- [x] Add a local batch generation CLI scaffold for the plain `T1 <-> T2` baseline.
- [x] Add a local train smoke CLI scaffold for the plain `T1 <-> T2` baseline.
- [x] Add a fast-validation `64x64` training configuration for quick loss checks.
- [x] Add checkpoint-backed intermediate activation extraction and visualization.
- [x] Run a local preprocessing smoke test for the plain `T1 <-> T2` baseline.
- [x] Run a local end-to-end train/test smoke test for the plain `T1 <-> T2` baseline.
- [x] Measure batch time, epoch time, and VRAM use for the plain `T1 <-> T2` baseline.
- [x] Estimate the expected duration of a full `200` epoch T1/T2 run on the local machine.
- [x] Extract the exact pseudo-RGB channel construction from `CycleGAN_batch1_RGB_CET1_T1_to_T2.ipynb`.
- [x] Identify whether `CycleGAN_T12FLAIR.ipynb` is baseline-critical or exploratory-only.
- [x] Define a whitelist for raw test inputs inside `data_set/BraTS2017TestingData/`.
- [x] Document explicitly that `BraTS2017TestingData/` is the generation-side test source, not yet the full downstream prognosis validation split.
- [x] Define a whitelist for raw training inputs inside `data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/`.
- [x] Decide whether 2D slices are generated on demand or cached to disk.
- [x] Define a metadata format for patient ID, modality, and slice index.
- [x] Define a first scriptable preprocessing scaffold from `.nii.gz` to derived paired 2D outputs.
- [x] Run a local smoke test on a very small subset.
- [x] Decide whether to keep the legacy `256` crop baseline unchanged for the first reproduction.
- [x] Add checkpoint and resume instructions for interrupted long runs.
- [x] Add checkpoint state saving and resume support for interrupted long runs.
- [x] Define the artifact layout for logs, checkpoints, samples, and test outputs.
- [ ] Extract the exact image-generation evaluation path for RMSE, Mutual Information, PSNR, and SSIM.
- [x] Add a first executable image-metric evaluation scaffold based on standard Python libraries.
- [x] Add a first executable NRRD-level evaluation scaffold for existing case artifacts.
- [x] Add a batch NRRD-level evaluation scaffold for all existing case artifacts.
- [x] Add a case-ranking scaffold to identify low-quality generations from batch metrics.
- [x] Audit the copied legacy checkpoint folder and record whether it is usable for inference/resume.
- [x] Extract the radiomics feature-extraction path from `Pre_operative(T2_radiomics).ipynb`.
- [x] Map the downstream chain from generated images to radiomics score to Lasso-Cox to Logrank evaluation.
- [x] Add a first executable PNG-to-NRRD reconstruction scaffold for downstream radiomics input.

Progress notes:

- legacy T1/T2 slice export has been identified as axial full-slice traversal
  with patient-and-slice-index pairing
- legacy stream mapping is `testA = T1`, `testB = T2`
- legacy volume channel references are currently recorded as `T1 -> 1`,
  `T2 -> 3`
- exact notebook export logic has been confirmed from
  `CycleGAN_batch1_RGB_T1_T2_.ipynb` as
  `np.flipud(img0[:, :, j, 1].T)` to `testA` and
  `np.flipud(img0[:, :, j, 3].T)` to `testB`
- `CycleGAN_T12FLAIR.ipynb` is now treated as supporting evidence for legacy
  conventions, not as the first reproduction target
- the prepare script under `scripts/prepare/` now performs paired NIfTI loading,
  axial slice traversal, optional PNG export, and manifest generation
- raw input whitelists, generation-side test terminology, and manifest metadata
  fields are now captured in `docs/2D/spec.md` and `configs/2d/dataset_t1_t2.yaml`
- clean local names for derived data, runs, generated outputs, metrics, and
  radiomics outputs are now captured in `docs/2D/spec.md`
- `uv run` execution is now wired through `pyproject.toml`
- the current preprocessing policy is raw-volume canonical input plus optional
  disposable PNG cache output
- smoke tests completed on `BraTS2017TestingData/` with `limit_cases=1`
  for both manifest-only and PNG-export paths
- a matching manifest-only smoke test also completed on
  `Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/` with `limit_cases=1`
- the first observed smoke-test case was `TCGA-02-0003` with `155` paired
  slices written into `A/` and `B/`
- a legacy-layout smoke test also completed with direct `testA/` and `testB/`
  export under `outputs/derived/t1_t2_cyclegan_legacy_smoke/`
- the first reproduction will keep the legacy `load_size=286` and
  `crop_size=256` settings
- checkpoint and resume guidance is now recorded in
  `configs/2d/train_t1_t2_cyclegan.yaml`
- pseudo-RGB `CE-T1 + T1 -> T2` construction is now recorded in
  `setting/cycle_gan_rgb_cet1_t1_to_t2.md` as
  `red = CE-T1`, `green = plain T1`, `blue = zero`
- the repo still lacks the actual CycleGAN training and test codebase, so the
  preprocessing path is runnable but the full baseline remains blocked on that
  external dependency
- the downstream radiomics-side evaluation file flow is now recorded in
  `setting/evaluation.md`, `configs/2d/eval_t1_t2.yaml`, and
  `docs/2D/runbook.md`
- user recollection now confirms that the image metrics were based on standard
  Python library implementations, with `scikit-learn` as the main baseline
  reference
- a first executable PNG-pair evaluation scaffold now exists under
  `scripts/eval/t1_t2_image_metrics.py`
- a first executable NRRD-level evaluation scaffold now exists under
  `scripts/eval/t1_t2_nrrd_metrics.py`
- a batch NRRD-level evaluation scaffold now exists under
  `scripts/eval/t1_t2_nrrd_metrics_batch.py`
- a case-ranking scaffold now exists under
  `scripts/eval/t1_t2_case_rankings.py`
- the copied legacy checkpoint folder has now been audited:
  discriminator weights and option logs are present, but generator weights are
  missing
- a first executable PNG-to-NRRD reconstruction scaffold now exists under
  `scripts/postprocess/reconstruct_t2_volumes.py`
- `scripts/clasical/` is now treated as reference-only rather than the intended
  local runtime path
- a local `src/xai_generics/` package now exists with:
  - paired T1/T2 dataset loading from raw `.nii.gz`
  - dataset inspection CLI
  - train CLI with epoch loop, checkpointing, and sample writing
  - local CycleGAN/checkpoint scaffold
  - inference CLI with checkpoint-backed output writing
  - batch generation CLI for generation-side test set exports
  - smoke train CLI scaffold
  - checkpoint-backed inference output writer
  - benchmark CLI for train/infer step timing
- local checkpoint-backed inference now writes paired `real_B` / `fake_B`
  PNGs and an inference manifest from the smoke checkpoint
- local benchmark runs now report average step time and an epoch-time estimate
  for both train and infer modes
- the generated smoke output also passes through the paired PNG metric
  evaluator without format changes
- a one-epoch local training run now saves `latest` and per-epoch checkpoints
  and writes sample outputs under `outputs/runs/t1_t2_cyclegan_2d/samples/`
- local inference now defaults to the generation-side test root
  `data_set/BraTS2017TestingData/`
- local GPU verification completed on `NVIDIA GeForce RTX 5070 Ti`
- current GPU benchmark is approximately `0.46s` per train step and
  `0.78s` per infer step in this environment, with CUDA memory usage reported
  by PyTorch
- rough `200`-epoch estimates are now recorded from the benchmark CLI output
- batch generation now writes paired PNGs plus `generation_manifest.csv` from
  the generation-side test root
- training logs are now appended to CSV, saved alongside checkpoint state, and
  can be plotted from `scripts/eval/plot_t1_t2_training_loss.py`
- fast-validation training now works at `64x64` with reduced `ngf/ndf` and
  fewer residual blocks
- a two-epoch short run on 6 slices completed and produced a loss plot under
  `outputs/runs/t1_t2_cyclegan_2d_fast_6s/logs/training_loss.png`
- fast-validation benchmark measurements are now available:
  about `0.37s` per train step and `0.17s` per infer step
- epoch-wise generation-side validation now writes `validation_log.csv`
  and `validation_metrics.png`
- checkpoint-backed intermediate activations can now be extracted as `.npy`
  and preview PNGs from arbitrary generator modules

## Task Order

1. Freeze the T1/T2 baseline assumptions.
2. Freeze T1/T2 hyperparameters.
3. Script T1/T2 preprocessing.
4. Run T1/T2 smoke test.
5. Run T1/T2 benchmark.
6. Launch the full T1/T2 baseline only after timing is understood.
7. Move to CE-T1/T1 and FLAIR-related variants later.

## Notes

- Training data is currently assumed to be `data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/`.
- Test data is currently assumed to be `data_set/BraTS2017TestingData/` because it explicitly includes `TestingData` in the directory name.
- Existing generated artifacts in the test directory must not be confused with canonical raw inputs.
- The first implementation should optimize for reproducibility, not for novelty.
- Downstream prognosis analysis exists in the legacy workflow, but the first implementation milestone is still the image-translation baseline itself.
