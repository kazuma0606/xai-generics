# CycleGAN T1/T2

Source notebook:

- `ipynb/CycleGAN_batch1_RGB_T1_T2_.ipynb`

Role:

- primary 2D baseline for the project
- first local reproduction target

Observed slice-generation logic from the legacy notebooks:

- source files are enumerated with:
  - `BRATS_*.nii.gz`
- loaded with:
  - `nibabel`
- observed volume shape comment:
  - `(240, 240, 155, 4)`
- slice traversal:
  - iterate all axial slices with `for j in range(img0.shape[2])`
- source image export:
  - `testA/<patient>_img_<slice>.png`
  - data expression: `np.flipud(img0[:, :, j, 1].T)`
- target image export:
  - `testB/<patient>_img_<slice>.png`
  - data expression: `np.flipud(img0[:, :, j, 3].T)`

Interpretation:

- `testA` is the T1-side input stream
- `testB` is the T2-side target stream
- pairing is by patient prefix and slice index
- the legacy 2D path uses all slices along the third axis

Evidence note:

- the explicit extraction snippet appears clearly in
  `ipynb/CycleGAN_T12FLAIR.ipynb`
- it is used here as the concrete legacy reference for the T1/T2 slice export
  pattern

Observed training settings:

- `dataroot: ./datasets/T1W2T2W`
- `batch_size: 1`
- `direction: AtoB`
- `crop_size: 256`
- `load_size: 286`
- `preprocess: resize_and_crop`
- `input_nc: 3`
- `output_nc: 3`
- `netG: resnet_9blocks`
- `netD: basic`
- `ngf: 64`
- `ndf: 64`
- `pool_size: 50`
- `display_freq: 400`
- `print_freq: 100`
- `save_latest_freq: 5000`
- `n_epochs: 100`
- `n_epochs_decay: 100`

Observed run names and resume points:

- `name: CE2RGB_T1_T2_FLIR_batch_1_cyclegan`
- resume example:
  - `epoch_count: 10`

Observed legacy checkpoint audit:

- checkpoint root:
  - `checkpoint/T1W2T2W_batch_1_cyclegan`
- legacy script entrypoints now available in this repo:
  - `scripts/clasical/train.py`
  - `scripts/clasical/test.py`
- `train_opt.txt` shows:
  - `continue_train: True`
  - `epoch: latest`
  - `epoch_count: 199`
- `test_opt.txt` shows:
  - `direction: BtoA`
  - `epoch: 200`
- discriminator checkpoints are present
- generator checkpoints are not present in the current copied folder

Practical conclusion:

- this checkpoint folder is useful as a settings and artifact reference
- it is not sufficient by itself for generator inference or training resume
  because `net_G_A` and `net_G_B` weights are missing

Observed command fragments:

- `python train.py --dataroot ./datasets/T1W2T2W --name CE2RGB_T1_T2_FLIR_batch_1_cyclegan --model cycle_gan --batch_size 1`
- `python train.py --dataroot ./datasets/T1W2T2W --name CE2RGB_T1_T2_FLIR_batch_1_cyclegan --model cycle_gan --batch_size 1 --continue_train --epoch_count 10`

Notes:

- the notebook filename suggests T1/T2, but the recorded run name includes
  `CE2RGB_T1_T2_FLIR`, which indicates historical naming drift
- for implementation, the modality priority should still be interpreted as
  plain `T1 <-> T2`
- this notebook remains the canonical source for the first translation baseline
