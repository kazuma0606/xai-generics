# 2D Configs

These are implementation-facing config templates derived from the legacy
notebooks and the `setting/` notes.

Current scope:

- primary baseline: `T1 <-> T2`
- local execution first
- 2D slice-wise pipeline

Files:

- `dataset_t1_t2.yaml`
- `prepare_t1_t2.yaml`
- `prepare_t1_t2_legacy_cyclegan.yaml`
- `reconstruct_t2_volumes.yaml`
- `train_t1_t2_cyclegan.yaml`
- `eval_t1_t2.yaml`
- `radiomics_t2.yaml`

Notes:

- these are templates, not yet wired to code
- raw `.nii.gz` volumes are treated as canonical inputs
- generated artifacts and notebook exports are downstream products
- `prepare_t1_t2.yaml` is the execution-facing reference for
  `scripts/prepare/t1_t2_2d_prepare.py`
- `prepare_t1_t2_legacy_cyclegan.yaml` is the legacy layout variant for
  external `pytorch-CycleGAN-and-pix2pix` style `trainA/trainB/testA/testB`
  dataroots
- `eval_t1_t2.yaml` now also maps to `scripts/eval/t1_t2_image_metrics.py`
  for a first reproducible image-metric reimplementation
- `reconstruct_t2_volumes.yaml` maps to the PNG-to-NRRD bridge used before
  downstream radiomics extraction
