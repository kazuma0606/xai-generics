# 2D Baseline Plan

## Goal

Build a reproducible 2D baseline implementation that replaces the current
notebook-only workflow while preserving the legacy experiment intent.

The first target is specifically the plain `T1 <-> T2` baseline informed by the
paper in `report/` and the matching legacy notebooks.

The word `test` in this plan refers first to generation/inference inputs for the
trained image translation model, not yet to the final downstream prognosis
evaluation split.

## Plan Summary

Phase A focuses on freezing assumptions.
Phase B focuses on data pipeline replacement.
Phase C focuses on a minimal runnable local baseline.
Phase D focuses on validation and cleanup.

## Phase A: Freeze the Baseline

Objectives:

- map notebook experiments to named baseline variants
- fix dataset root definitions
- fix baseline hyperparameters
- identify which notebook outputs are source data and which are artifacts

Deliverables:

- baseline spec document
- explicit mapping of:
  - `T1 <-> T2` as the primary baseline
  - `CE-T1 + T1 -> T2` as a secondary extension
  - optional FLAIR-related exploratory branch

## Phase B: Replace Notebook Data Prep

Objectives:

- move from ad hoc notebook conversion to scripted preprocessing
- use `.nii.gz` as canonical input
- generate deterministic 2D slices or cached derived slices
- start with T1/T2 pairing only

Required decisions:

- axial-only or other view support
- all-slice vs filtered-slice policy
- naming convention for paired samples
- whether cached 2D images are written to disk or generated on demand

Preferred implementation direction:

- scriptable preprocessing
- metadata-first indexing
- optional cache layer

Decision now recorded:

- raw `.nii.gz` remains canonical
- cached 2D PNG export is optional and disposable
- the first smoke tests have validated both manifest-only and PNG-cache paths

## Phase C: Local Runnable Baseline

Objectives:

- run a local reimplementation of the legacy CycleGAN baseline on the RTX 5070 Ti
- verify memory usage and wall-clock throughput
- avoid immediately committing to full `200` epoch runs
- do this first for the plain T1/T2 baseline

Important implementation boundary:

- `scripts/clasical/` is reference-only and not the intended runtime path
- the executable baseline should be implemented natively inside this repository
- the old checkpoint folder is configuration evidence, not the primary runtime
  dependency

Execution strategy:

1. run a smoke test
2. run a short benchmark training segment
3. estimate full-run duration from measured throughput
4. only then start a long baseline training run

Recommended first checkpoints:

- dataset indexing works
- one batch can be loaded
- one forward/backward pass succeeds
- one epoch or fixed-iteration run finishes
- interrupted runs can be resumed with explicit checkpoint state and
  `continue_train` plus `epoch_count` style settings

Required local components:

- dataset manifest reader or direct NIfTI-backed dataset loader
- T1/T2 paired 2D slice dataset class
- CycleGAN-compatible model wrapper
- train CLI
- inference CLI
- checkpoint save/load that explicitly preserves generator and discriminator
  weights
- artifact writing for generated slices and run metadata
- hooks into the existing evaluation and radiomics bridge scripts

## Phase D: Validation

Objectives:

- compare outputs against notebook expectations
- confirm modality direction and channel construction are correct
- confirm train/test split is not mixed
- confirm generation metrics can be reproduced

Validation items:

- sample visualization
- patient/slice pairing sanity checks
- output directory structure
- checkpoint generation
- run log consistency
- RMSE / Mutual Information / PSNR / SSIM reproduction path

## Downstream Continuation

After the 2D `T1 <-> T2` baseline is stable, the next chain can extend into:

- radiomics feature extraction from real and generated outputs
- radiomics score construction
- Lasso-Cox based feature selection
- Logrank-based prognosis group comparison

This continuation is informed by:

- `ipynb/Pre_operative(T2_radiomics).ipynb`
- `ipynb/rad_score_T2_to_FLAIR.ipynb`

## Risks

### Risk 1: Hidden Notebook Logic

The notebooks may contain implicit preprocessing choices that are not obvious
from the train command alone.

Mitigation:

- extract conversion cells explicitly
- document assumptions instead of silently reproducing them

### Risk 2: Test Data Pollution

`BraTS2017TestingData/` contains both raw and derived artifacts.

Mitigation:

- whitelist raw `.nii.gz`
- ignore `fake_*`, `real_*`, `CSV/`, and notebook-generated PNG outputs as
  canonical inputs

### Risk 3: Local Runtime Still Long

Even on the local GPU, the original baseline may still take days.

Mitigation:

- benchmark first
- add resumable checkpoints
- separate smoke, benchmark, and full-train modes

### Risk 4: Overcommitting to 3-Channel Legacy Format

The long-term project direction is not limited to RGB-like inputs.

Mitigation:

- treat 3-channel formatting as a compatibility baseline only
- isolate channel-construction logic so it can later be swapped for N-channel

## Immediate Next Step

Implement the minimum local data and run path for the plain `T1 <-> T2` 2D
baseline first. The pseudo-RGB and FLAIR-related variants should follow only
after the plain baseline path is reproducible.
