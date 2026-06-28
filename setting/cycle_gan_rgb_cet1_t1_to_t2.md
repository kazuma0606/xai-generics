# CycleGAN RGB CE-T1 + T1 -> T2

Source notebook:

- `ipynb/CycleGAN_batch1_RGB_CET1_T1_to_T2.ipynb`

Role:

- secondary 2D baseline
- pseudo-RGB extension after the plain T1/T2 path is stable

Observed training settings:

- `dataroot: ./datasets/T1W2T2W/RGB_CET1-T1_to_T2`
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

Observed pseudo-RGB construction:

- source result directory:
  - `results/<case>/T1W2CE_batch_1_cyclegan/test_200/images`
- plain T1 source images are read from:
  - `*_fake_A.png`
- CE-T1 source images are read from:
  - `*_real_A.png`
- each loaded image is resized to:
  - `(240, 240)`
- channel extraction logic:
  - plain T1 contributes the green channel
  - CE-T1 contributes the red channel
  - blue channel is a zero-filled black image
- exact merge expression observed:
  - `Image.merge('RGB', (CE_T1_r[i], plain_T1_g[i], pil_image))`
- export target:
  - `datasets/T1W2T2W/RGB_CET1-T1_to_T2/trainB/<patient>_RGB_CET1-T1_img_<slice>.png`

Observed run names and resume points:

- `name: RGB_CET1-T1_to_T2_batch_1_cyclegan`
- resume example:
  - `epoch_count: 141`

Observed command fragments:

- `python train.py --dataroot ./datasets/T1W2T2W/RGB_CET1-T1_to_T2 --name RGB_CET1-T1_to_T2_batch_1_cyclegan --model cycle_gan --batch_size 1`
- `python train.py --dataroot ./datasets/T1W2T2W/RGB_CET1-T1_to_T2 --name RGB_CET1-T1_to_T2_batch_1_cyclegan --model cycle_gan --batch_size 1 --continue_train --epoch_count 141`

Notes:

- the notebook explicitly uses pseudo-RGB style multi-modal input
- this should not replace the plain T1/T2 baseline as the first target
