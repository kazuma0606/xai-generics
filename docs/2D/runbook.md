# 2D Baseline Runbook

## Current State

This repository now contains:

- baseline assumptions and task tracking under `docs/2D/`
- notebook-derived settings under `setting/`
- execution-facing config templates under `configs/2d/`
- a working NIfTI-to-2D preparation script under `scripts/prepare/`
- a first image-metric evaluation scaffold under `scripts/eval/`
- a first PNG-to-NRRD reconstruction scaffold under `scripts/postprocess/`
- legacy CycleGAN entry scripts under `scripts/clasical/`
- a local `src/xai_generics/` package for the new runtime path

This repository does not yet contain the actual CycleGAN training/test codebase
such as the full legacy `pytorch-CycleGAN-and-pix2pix` project structure
(`models/`, `options/`, `util/`, dataset classes, and the rest of the package).

That means the plain `T1 <-> T2` baseline is fully specified and partially
scripted here, but not yet runnable end-to-end from this repository alone.

## What Already Runs

Confirmed local commands:

- `$env:UV_CACHE_DIR='C:\Users\yoshi\xai-generics\.uv-cache'`
- `uv run python scripts/prepare/t1_t2_2d_prepare.py --input-root data_set/BraTS2017TestingData --output-root outputs/derived/t1_t2_2d --split-name generation_test_smoke --limit-cases 1`
- `uv run python scripts/prepare/t1_t2_2d_prepare.py --input-root data_set/BraTS2017TestingData --output-root outputs/derived/t1_t2_2d --split-name generation_test_smoke_png --limit-cases 1 --write-png`
- `uv run python scripts/prepare/t1_t2_2d_prepare.py --input-root data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations --output-root outputs/derived/t1_t2_2d --split-name train_smoke --limit-cases 1`
- `uv run python scripts/prepare/t1_t2_2d_prepare.py --input-root data_set/BraTS2017TestingData --output-root outputs/derived/t1_t2_cyclegan_legacy_smoke --split-name generation_test --limit-cases 1 --write-png --flat-output`
- `uv run python scripts/eval/t1_t2_image_metrics.py --image-root outputs/smoke_eval_case/images --output-dir outputs/smoke_eval_case/metrics`
- `uv run python scripts/postprocess/reconstruct_t2_volumes.py --image-root outputs/smoke_eval_case/images --output-dir outputs/smoke_eval_case/radiomics`
- `uv run python scripts/eval/t1_t2_nrrd_metrics.py --real-volume data_set/BraTS2017TestingData/TCGA-02-0003/real_T2.nrrd --fake-volume data_set/BraTS2017TestingData/TCGA-02-0003/fake_T2_epoch_200.nrrd --output-dir outputs/metrics/t1_t2_cyclegan_2d/TCGA-02-0003_nrrd`
- `uv run python -m xai_generics.cli.inspect_t1_t2 --limit 2`
- `uv run python -m xai_generics.cli.train_t1_t2 --dry-run`
- `uv run python -m xai_generics.cli.train_t1_t2 --epochs 1 --max-steps 2`
- `uv run python -m xai_generics.cli.train_t1_t2 --epochs 1 --max-steps 1 --checkpoint-tag latest`
- `uv run python -m xai_generics.cli.train_t1_t2 --epochs 2 --max-steps 3 --checkpoint-tag latest --resume --resume-tag latest`
- `uv run python -m xai_generics.cli.train_t1_t2 --train-config configs/2d/train_t1_t2_cyclegan_fast.yaml --dataset-config configs/2d/dataset_t1_t2_fast.yaml --epochs 2 --max-samples 6 --checkpoint-tag latest --run-root outputs/runs/t1_t2_cyclegan_2d_fast_6s --checkpoint-root outputs/runs/t1_t2_cyclegan_2d_fast_6s/checkpoints --sample-root outputs/runs/t1_t2_cyclegan_2d_fast_6s/samples --log-root outputs/runs/t1_t2_cyclegan_2d_fast_6s/logs`
- `uv run python -m xai_generics.cli.train_t1_t2 --train-config configs/2d/train_t1_t2_cyclegan_fast.yaml --dataset-config configs/2d/dataset_t1_t2_fast.yaml --epochs 1 --max-samples 3 --validate-every-epoch --validation-limit 3 --checkpoint-tag latest --run-root outputs/runs/t1_t2_cyclegan_2d_fast_val --checkpoint-root outputs/runs/t1_t2_cyclegan_2d_fast_val/checkpoints --sample-root outputs/runs/t1_t2_cyclegan_2d_fast_val/samples --log-root outputs/runs/t1_t2_cyclegan_2d_fast_val/logs --validation-output-root outputs/runs/t1_t2_cyclegan_2d_fast_val/validation`
- `uv run python -m xai_generics.cli.infer_t1_t2 --dry-run`
- `uv run python -m xai_generics.cli.smoke_train_t1_t2 --dry-run`
- `uv run python -m xai_generics.cli.infer_t1_t2 --checkpoint-tag smoke --limit 2 --output-dir outputs/generated/t1_t2_cyclegan_2d/smoke_test`
- `uv run python -m xai_generics.cli.infer_t1_t2 --checkpoint-tag latest --limit 1 --output-dir outputs/generated/t1_t2_cyclegan_2d/latest_test`
- `uv run python -m xai_generics.cli.generate_t1_t2 --limit 2 --output-dir outputs/generated/t1_t2_cyclegan_2d/batch_smoke`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --mode train --warmup 0 --steps 2`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --mode infer --warmup 0 --steps 2`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --mode train --warmup 0 --steps 1`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --mode infer --warmup 0 --steps 1`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --train-config configs/2d/train_t1_t2_cyclegan_fast.yaml --dataset-config configs/2d/dataset_t1_t2_fast.yaml --mode train --warmup 0 --steps 1`
- `uv run python -m xai_generics.cli.benchmark_t1_t2 --train-config configs/2d/train_t1_t2_cyclegan_fast.yaml --dataset-config configs/2d/dataset_t1_t2_fast.yaml --mode infer --warmup 0 --steps 1`
- `uv run python scripts/eval/t1_t2_image_metrics.py --image-root outputs/generated/t1_t2_cyclegan_2d/smoke_test_fix --output-dir outputs/generated/t1_t2_cyclegan_2d/smoke_test_fix/metrics`
- `uv run python scripts/eval/t1_t2_image_metrics.py --image-root outputs/generated/t1_t2_cyclegan_2d/latest_test --output-dir outputs/generated/t1_t2_cyclegan_2d/latest_test/metrics`
- `uv run python scripts/eval/plot_t1_t2_training_loss.py --log-path outputs/runs/t1_t2_cyclegan_2d/logs/training_log.csv --output-path outputs/runs/t1_t2_cyclegan_2d/logs/training_loss.png`
- `uv run python scripts/eval/plot_t1_t2_training_loss.py --log-path outputs/runs/t1_t2_cyclegan_2d_fast_6s/logs/training_log.csv --output-path outputs/runs/t1_t2_cyclegan_2d_fast_6s/logs/training_loss.png`
- `uv run python scripts/eval/plot_t1_t2_validation_metrics.py --log-path outputs/runs/t1_t2_cyclegan_2d_fast_val/validation/validation_log.csv --output-path outputs/runs/t1_t2_cyclegan_2d_fast_val/validation/validation_metrics.png`
- `uv run python -m xai_generics.cli.extract_t1_t2_activation --checkpoint-dir outputs/runs/t1_t2_cyclegan_2d_fast_10s/checkpoints --checkpoint-tag latest --side A --list-modules`
- `uv run python -m xai_generics.cli.extract_t1_t2_activation --checkpoint-dir outputs/runs/t1_t2_cyclegan_2d_fast_10s/checkpoints --checkpoint-tag latest --side A --module-path 6 --sample-index 0 --output-dir outputs/activations/t1_t2_cyclegan_fast_10s`

Observed smoke-test outputs:

- first generation smoke-test case: `TCGA-02-0003`
- observed slice count: `155`
- paired exports written under `A/` and `B/`
- manifest written successfully
- direct legacy-compatible `testA/` and `testB/` export also works
- evaluation smoke test completed on `3` synthetic pairs
- reconstruction smoke test completed with output volume shape `(3, 8, 8)`
- direct NRRD evaluation completed on `TCGA-02-0003` across `155` slices
- local `src` dataset inspection completed on the training root with `15810`
  paired slices discovered
- local `src` train CLI dry run completed successfully
- local `src` inference CLI dry run completed successfully
- local `src` smoke-train CLI dry run completed successfully
- local `src` inference smoke run completed successfully and wrote paired
  `real_B` / `fake_B` PNG outputs from the `smoke` checkpoint
- local `src` benchmark runs completed for both `train` and `infer` modes with
  2 timed steps each
- the generated smoke output was verified by the image-metric evaluator with
  one paired slice
- the `latest` checkpoint from the local train CLI also produced paired PNGs
  and was verified by the image-metric evaluator
- checkpoint state saving and resume now work through `latest_state.pt`
- loss history is append-only in `training_log.csv` and can be plotted to
  `training_loss.png`
- fast-validation config completed a 2 epoch, 6-slice run at `64x64`
  resolution
- fast-validation benchmark measurements reported about `0.3735s` per train
  step and `0.1714s` per infer step
- epoch-wise validation completed on the generation-side test root and wrote
  `validation_log.csv` plus `validation_metrics.png`
- checkpoint-backed intermediate activation extraction now works for generator
  modules and writes `.npy`, preview PNGs, and a JSON summary
- local inference and infer benchmarking now default to
  `data_set/BraTS2017TestingData/` as the generation-side test source
- batch generation from the generation-side test root now works through
  `src/xai_generics/cli/generate_t1_t2.py`
- the batch smoke output was verified by the image-metric evaluator with
  2 paired slices
- `uv run` now resolves to CUDA-enabled PyTorch:
  `torch 2.11.0+cu128`, `cuda_available=True`, `device_name=NVIDIA GeForce RTX 5070 Ti`
- the current benchmark samples report approximately `0.4576s` per train step
  and `0.7768s` per infer step, with CUDA memory usage reported by PyTorch
- extrapolated `200`-epoch estimates from those benchmark samples are roughly
  `401.94` hours for train and `220.75` hours for infer
- benchmark output in this environment reported `peak_cuda_allocated_mb=n/a`
  because CUDA was not available in the execution environment

## End-to-End Target Flow

The intended first local baseline flow is:

1. prepare paired 2D T1/T2 slices from raw `.nii.gz`
2. train CycleGAN with the legacy `batch_size=1`, `load_size=286`,
   `crop_size=256` settings
3. generate test outputs for the generation-side test source
4. compute image-level metrics
5. reconstruct or consume volume-style outputs for downstream radiomics
6. extract radiomics features and later continue to prognosis modeling

## External Dependency Gap

To make the baseline fully runnable, the following external code is still
required:

- CycleGAN training entrypoint compatible with the notebook commands
- CycleGAN test/inference entrypoint compatible with the notebook commands
- dataset loader wiring for the prepared `A/` / `B/` layout
- artifact writing for checkpoints, logs, generated images, and results

The historical notebooks point to the legacy external project path:

- `pytorch-CycleGAN-and-pix2pix`

Bridge note:

- see `docs/2D/legacy_cyclegan_bridge.md` for the expected handoff layout and
  commands

Legacy script note:

- `scripts/clasical/train.py` and `scripts/clasical/test.py` are now present
- they match the standard top-level entry scripts from the old CycleGAN code
- the rest of that codebase is still missing, so these files alone are not yet
  executable here

## Legacy Evaluation Path Observed

The notebook evidence currently supports this downstream path:

1. generated images are written under a path like:
   `results/<case>/T1W2T2W_batch_1_cyclegan/test_200/images`
2. per-slice generated outputs are read from image patterns such as:
   `*_<slice>_real_B.png`
3. case-level scalar volumes can be reconstructed from the PNG slices
4. downstream radiomics notebooks consume volume-style files:
   - `real_T2.nrrd`
   - `fake_T2_epoch_200.nrrd`
   - `contour.nrrd`
5. those are loaded as `(155, 240, 240)` arrays in the legacy notebook
6. PyRadiomics is then run with `contour.nrrd` as the mask and CSV files are
   written:
   - `CSV/real_T2.csv`
   - `CSV/fake_T2.csv`

The exact image-level RMSE / Mutual Information / PSNR / SSIM implementation is
still not isolated in the notebooks and remains a pending extraction item.

Current executable fallback:

- `scripts/eval/t1_t2_image_metrics.py` now provides a standard-library-based
  reimplementation for paired `real_B` / `fake_B` PNG evaluation
- `scripts/eval/t1_t2_nrrd_metrics.py` now provides a standard-library-based
  reimplementation for directly evaluating existing `real_T2.nrrd` and
  `fake_T2_epoch_200.nrrd` case artifacts
- `scripts/eval/t1_t2_nrrd_metrics_batch.py` now provides a batch wrapper for
  aggregating NRRD-level metrics across all case directories
- `scripts/eval/t1_t2_case_rankings.py` now provides a quick way to identify
  worst-case and low-similarity cases from the aggregated case summary
- `scripts/postprocess/reconstruct_t2_volumes.py` now provides a first
  PNG-to-NRRD reconstruction bridge for `real_T2.nrrd` and
  `fake_T2_epoch_200.nrrd`
- this is intended as the first reproducible evaluation path even if the exact
  historical notebook cell is not recovered

Current verification caveat:

- `uv run` verification now works after dependency sync and by fixing
  `UV_CACHE_DIR` inside the repository
- the earlier issue was the default global uv cache path, not a project-level
  limitation of `uv` or `.venv`
- the checked-in `BraTS2017TestingData/<case>/T1` and `T2` directories appear
  to be legacy paired reference caches rather than the raw `_real_B/_fake_B`
  CycleGAN image dump naming
- the copied checkpoint folder `checkpoint/T1W2T2W_batch_1_cyclegan` contains
  discriminator weights and option logs, but no generator weights, so it cannot
  be used directly for inference or full training resume
- the `src` runtime path now has a smoke-train scaffold, but the real PyTorch
  training loop still needs the `torch` dependency to be installed before it can
  execute

## Next Practical Step

The next blocking decision is not in preprocessing anymore.

One of these needs to happen next:

- bring the legacy CycleGAN code into this workspace, or
- write a new local training/inference implementation that matches the notebook
  assumptions closely enough for the first reproduction
