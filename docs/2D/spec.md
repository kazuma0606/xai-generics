# 2D Baseline Spec

## Purpose

This document defines the first reproducible 2D baseline for this repository.
The target is not a new architecture. The target is to reproduce and stabilize
the legacy CycleGAN workflow from the notebooks in a form that can be rerun on
the local machine.

## Scope

In scope:

- 2D slice-wise CycleGAN baseline
- T1 <-> T2 translation baseline
- pseudo-RGB style CE-T1 + T1 -> T2 variant only after the T1/T2 baseline is stable
- local execution on the current workstation
- dataset preparation from raw `.nii.gz` volumes
- reproducible train/test split based on the existing `data_set/` layout

Out of scope for this phase:

- 2.5D and 3D models
- Lightning refactor
- radiomics and prognosis modeling
- XAI evaluation
- cloud execution

## Source Assets

Primary references:

- `PROJECT_OVERVIEW.md`
- `report/13246_2024_Article_1443 (1).pdf`
- `ipynb/CycleGAN_batch1_RGB_T1_T2_.ipynb`
- `ipynb/CycleGAN_batch1_RGB_CET1_T1_to_T2.ipynb`
- `ipynb/CycleGAN_T12FLAIR.ipynb`

Current baseline priority:

- use `T1` and `T2` as the primary data sources
- treat the T1/T2 setting as the first 2D reproduction target
- defer CE-T1 and FLAIR extensions until after the plain T1/T2 path works

## Dataset Definition

Training source:

- `data_set/Pre-operative_TCGA_GBM_NIfTI_and_Segmentations/`
- directory count observed: `102`
- each case contains at least:
  - `*_t1.nii.gz`
  - `*_t1Gd.nii.gz`
  - `*_t2.nii.gz`
  - `*_flair.nii.gz`
  - segmentation volumes
- observed segmentation filenames include examples such as:
  - `*_GlistrBoost.nii.gz`
  - `*_GlistrBoost_ManuallyCorrected.nii.gz`

Test source:

- `data_set/BraTS2017TestingData/`
- directory count observed: `33`
- this is the generation-side test dataset because the directory name includes `TestingData`

Terminology note:

- `training` means data used to fit the image translation model
- `test` here means data used to run the trained model and generate synthetic
  outputs
- this `test` term does not yet mean a complete end-to-end prognosis validation
  split for downstream survival modeling

Important note:

- `BraTS2017TestingData/` already contains generated artifacts such as `.png`,
  `.nrrd`, `fake_*`, `real_*`, and `CSV/`.
- The 2D baseline must distinguish raw source volumes from derived artifacts.
- For reproducibility, raw `.nii.gz` files are the canonical source. Existing
  generated `.png` outputs are reference artifacts, not authoritative inputs.

Current baseline modality priority:

- primary source modalities: `T1` and `T2`
- the first 2D image-translation baseline is `T1 <-> T2`
- downstream radiomics and prognosis work may later consume real and generated
  `T2`-side outputs

## Canonical 2D Data Policy

The canonical storage format is raw volume data, not permanently expanded 2D
PNG datasets.

Rules:

- keep `.nii.gz` as the source of truth
- do not treat notebook-generated PNG dumps as the only training source
- generate 2D slices deterministically from `.nii.gz`
- make slice selection rules explicit and versioned
- if caching 2D outputs is needed, treat them as disposable derived artifacts
- every preparation run should emit a manifest file

Rationale:

- raw volumes are smaller and easier to version conceptually
- notebook-style full PNG expansion increases disk usage and duplicates logic
- direct slicing from `.nii.gz` keeps preprocessing consistent

## Baseline Model Definition

Initial baseline from notebooks:

- model: `cycle_gan`
- generator: `resnet_9blocks`
- discriminator: `basic`
- `batch_size: 1`
- `load_size: 286`
- `crop_size: 256`
- `preprocess: resize_and_crop`
- `input_nc: 3`
- `output_nc: 3`
- `ngf: 64`
- `ndf: 64`
- `n_epochs: 100`
- `n_epochs_decay: 100`
- total nominal training length: `200` epochs

Observed notebook commands:

- `T1W2T2W` baseline:
  - `python train.py --dataroot ./datasets/T1W2T2W --name T1W2T2W_batch_1_cyclegan --model cycle_gan --batch_size 1`
- `RGB_CET1-T1_to_T2` baseline:
  - `python train.py --dataroot ./datasets/T1W2T2W/RGB_CET1-T1_to_T2 --name RGB_CET1-T1_to_T2_batch_1_cyclegan --model cycle_gan --batch_size 1`

## Image Generation Evaluation

Legacy evaluation uses image similarity metrics on generated outputs.

Confirmed metrics from the existing workflow:

- RMSE
- Mutual Information
- PSNR
- SSIM

These belong to the image-generation evaluation stage, before downstream
radiomics or prognosis analysis.

CycleGAN direction convention for the first baseline:

- domain `A = T1`
- domain `B = T2`
- `AtoB` means `real_A -> fake_B`, so T1 input generates synthetic T2
- `BtoA` means `real_B -> fake_A`, so T2 input generates synthetic T1

For the first `T1 -> T2` evaluation target:

- compare `real_B` against `fake_B`
- reconstruct downstream `real_T2` / `fake_T2` artifacts from the `B` side

## Downstream Analysis Chain

The legacy notebooks indicate that image translation is not the terminal output.
The pipeline continues into radiomics and prognosis analysis.

Relevant notebooks:

- `ipynb/Pre_operative(T2_radiomics).ipynb`
- `ipynb/rad_score_T2_to_FLAIR.ipynb`

Observed downstream flow:

1. generate synthetic images from the translation model
2. reconstruct case-level real/generated T2 volumes for downstream use
3. extract radiomics features from real and generated images
4. compute radiomics-based scores
5. perform feature selection with Lasso-Cox style modeling
6. evaluate prognosis groups with Logrank testing

Important scope boundary:

- this document covers the 2D image-translation baseline first
- downstream radiomics/prognosis is part of the broader project chain, but is
  not required for the first runnable translation baseline

## Input Representation

Phase 1 baseline:

- preserve notebook-compatible 3-channel inputs for the legacy T1/T2 baseline
- use the T1/T2 notebook path first
- use pseudo-RGB only in a later secondary phase

Planned internal representation:

- read raw modalities from `.nii.gz`
- build the final 2D tensor in the dataset pipeline
- avoid hard-coding image generation logic inside notebooks

## Slice Extraction Rules

The exact notebook slice policy still needs to be codified, but the spec for the
first implementation should support these requirements:

- select slices per case deterministically
- preserve patient identity in filenames or metadata
- preserve modality identity
- allow train/test generation from separate roots
- support both direct T1/T2 pairing and CE-T1 + T1 -> T2 pairing

Pending decision:

- whether all axial slices are used
- whether empty or near-empty slices are filtered
- whether the segmentation mask is used to constrain slice selection

Until proven otherwise, the default baseline assumption is:

- axial 2D slices
- all eligible slices from each volume
- deterministic pairing by patient and slice index
- a derived dataset layout with source/target slice paths plus manifest metadata
- keep the first reproduction independent of segmentation masks
- revisit segmentation-based filtering only after the image-only baseline is stable

## Derived Dataset Layout

The prepared 2D dataset should support a scriptable derived layout such as:

```text
outputs/derived/t1_t2_2d/
  train/
    A/
    B/
    manifest.csv
  generation_test/
    A/
    B/
    manifest.csv
```

Where:

- `A` is the source-side stream for generation
- `B` is the target-side stream for paired evaluation
- `manifest.csv` records patient ID, slice index, source/target modalities,
  volume paths, and derived slice paths

The preparation script should also support the legacy flat CycleGAN layout when
needed:

```text
outputs/derived/t1_t2_cyclegan_legacy/
  trainA/
  trainB/
  testA/
  testB/
  train_manifest.csv
  generation_test_manifest.csv
```

Where:

- `trainA/trainB` map directly to the legacy `--dataroot` expectation for
  training
- `testA/testB` map directly to the legacy `--dataroot` expectation for
  generation-side testing
- manifests remain outside the image folders so slice provenance is still
  reproducible

## Naming Convention

The first 2D baseline should use stable names that separate documents,
preparation scripts, derived data, training runs, and evaluation artifacts.

Recommended convention:

- docs: `docs/2D/`
- notebook-derived reference notes: `setting/`
- execution-facing config templates: `configs/2d/`
- preprocessing scripts: `scripts/prepare/`
- derived paired dataset root: `outputs/derived/t1_t2_2d/`
- training run canonical name: `t1_t2_cyclegan_2d`
- generation output root: `outputs/generated/t1_t2_cyclegan_2d/`
- metric output root: `outputs/metrics/t1_t2_cyclegan_2d/`
- downstream radiomics output root: `outputs/radiomics/t1_t2_cyclegan_2d/`
- postprocess scripts: `scripts/postprocess/`

Legacy notebook run names should still be recorded when they differ from the
clean local naming convention, but new scriptable work should use the clean
names above.

## Artifact Layout

The first runnable baseline should write artifacts into predictable roots such
as:

```text
outputs/
  derived/
    t1_t2_2d/
      train/
      generation_test/
  runs/
    t1_t2_cyclegan_2d/
      checkpoints/
      logs/
      samples/
  generated/
    t1_t2_cyclegan_2d/
      generation_test/
  metrics/
    t1_t2_cyclegan_2d/
      generation_test/
  radiomics/
    t1_t2_cyclegan_2d/
      generation_test/
```

Where:

- `derived/` contains deterministic inputs prepared from raw `.nii.gz`
- `runs/` contains the live training workspace
- `checkpoints/` contains saved model states
- `logs/` contains training logs and benchmark notes
- `samples/` contains intermediate visual checks during training
- `generated/` contains model outputs produced from the generation-side test set
- `metrics/` contains RMSE, Mutual Information, PSNR, and SSIM summaries
- `radiomics/` contains downstream derived files such as `.nrrd` and `.csv`

## Runtime Expectation

Notebook logs imply the original baseline was long-running even in 2D.

Observed signal:

- training logs show roughly `0.34s` per iteration in long runs
- one run reached roughly `14,000` iterations in one epoch region of the log
- a full `200` epoch run can plausibly take multiple days to around 1-2 weeks

Implication:

- local reproduction should begin with short runs
- first target is correctness and throughput measurement, not full convergence

## Acceptance Criteria

The 2D baseline is considered established when:

- train/test dataset roots are explicitly defined in code and docs
- raw `.nii.gz` to 2D slice logic is reproducible
- one short local training run completes end-to-end
- produced artifacts are stored in a predictable location
- the legacy notebook settings can be mapped to CLI/config form
- a future full training run can be launched without notebook-only steps
- the plain `T1 <-> T2` baseline is runnable before any CE-T1 or FLAIR variant
