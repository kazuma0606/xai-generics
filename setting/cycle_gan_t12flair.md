# CycleGAN T1/T2/FLAIR

Source notebook:

- `ipynb/CycleGAN_T12FLAIR.ipynb`

Role:

- exploratory notebook with FLAIR-related extension
- useful reference for long-run behavior and evaluation commands

Observed training settings:

- `dataroot: ./datasets/T1W2T2W`
- `batch_size: 1`
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

- `name: T1W2T2W_batch_1_cyclegan`
- resume example:
  - `epoch_count: 199`

Observed command fragments:

- `python train.py --dataroot ./datasets/T1W2T2W --name T1W2T2W_batch_1_cyclegan --model cycle_gan --batch_size 1`
- `python train.py --dataroot ./datasets/T1W2T2W --name T1W2T2W_batch_1_cyclegan --model cycle_gan --batch_size 1 --continue_train --epoch_count 199`
- `python test.py --dataroot ./datasets/T1W2T2W --name T1W2T2W_batch_1_cyclegan --model cycle_gan --epoch 50 --direction BtoA`

Observed test-time settings:

- `batch_size: 1`
- `crop_size: 256`
- `load_size: 256`
- `preprocess: resize_and_crop`
- `direction: BtoA`

Notes:

- this notebook contains useful evidence for the long training duration of the
  legacy setup
- it should be treated as an extension/reference notebook, not the first
  implementation target
