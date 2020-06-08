# DeepDeblur-PyTorch

This is a pytorch implementation of our research. Please refer to our CVPR 2017 paper for details:

Deep Multi-scale Convolutional Neural Network for Dynamic Scene Deblurring
[[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nah_Deep_Multi-Scale_Convolutional_CVPR_2017_paper.pdf)]
[[supplementary](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Nah_Deep_Multi-Scale_Convolutional_2017_CVPR_supplemental.zip)]
[[slide](https://drive.google.com/file/d/1sj7l2tGgJR-8wTyauvnSDGpiokjOzX_C/view?usp=sharing)]

If you find our work useful in your research or publication, please cite our work:
```
@InProceedings{Nah_2017_CVPR,
  author = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```

Original Torch7 implementaion is available [here](https://github.com/SeungjunNah/DeepDeblur_release).

## Dependencies

* python 3 (tested with anaconda3)
* PyTorch 1.5
* tqdm
* imageio
* scikit-image
* numpy
* matplotlib
* readline
* (optional) [Apex](https://github.com/NVIDIA/apex) 0.1 (CUDA version should exactly match to PyTorch)

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# if the above installation fails, try: python setup.py --cpp_ext
```

## Datasets

* GOPRO_Large: [link](https://seungjunnah.github.io/Datasets/gopro)
* REDS: [link](https://seungjunnah.github.io/Datasets/reds)

## Usage examples

Put the datasets on a desired directory. By default, the data root is set as '~/Research/dataset'
See: src/option.py
```python
group_data.add_argument('--data_root', type=str, default='~/Research/dataset', help='dataset root location')
```

```bash
# single GPU training
python main.py --n_GPUs 1 --batch_size 8

# half precision testing (single precision for training, do not mix with mixed-precision training)
python main.py --n_GPUs 1 --batch_size 8 --precision half

# adversarial training
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+1*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+3*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+0.1*ADV

# train with GOPRO_Large dataset
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large
# train with REDS dataset (always set --do_test false)
python main.py --n_GPUs 1 --batch_size 8 --dataset REDS --do_test false --milestones 100 150 180 --endEpoch 200
```

```bash
# multi-GPU training (DataParallel)
python main.py --n_GPUs 2 --batch_size 16
```

```bash
# multi-GPU training (DistributedDataParallel), recommended
# single command version (do not set ranks)
python launch.py --n_GPUs 2 main.py --batch_size 16

# multi-command version (type in independent shells with the corresponding ranks)
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 0 # shell 0
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 1 # shell 1
```

```bash
# optional mixed-precision training (Apex required)
# mixed precision training may result in different accuracy
python main.py --n_GPUs 1 --batch_size 16 --amp true
python main.py --n_GPUs 2 --batch_size 16 --amp true
python launch.py --n_GPUs 2 main.py --batch_size 16 --amp true
```

```bash
# Advanced usage (recommended)
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --endEpoch 1000 --save_results none
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --endEpoch 1000 --save_results part
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --endEpoch 1000 --save_results all
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --endEpoch 1000 --save_results all --amp true

python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --endEpoch 200 --save_results all --do_test false
python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --endEpoch 200 --save_results all --do_test false --do_validate false
```

For more advanced usage, please take a look at src/option.py

## Results

* Single-precision training results

Dataset | GOPRO_Large | REDS
-- | -- | --
PSNR | 30.40 | 32.89
SSIM | 0.9018 | 0.9207
Download | [link](https://drive.google.com/file/d/1-wGC6s2D2ba-PSV60AeHf48HtYd9JkQ4/view?usp=sharing) | [link](https://drive.google.com/file/d/1aSPgVsNcPNqeGPn0Y2uGmgIwaIn5Njkv/view?usp=sharing)

<!-- * Mixed-precision training results

Dataset | GOPRO_Large | REDS
PSNR| -- | --
SSIM| -- | --
Model download | -- | --

Mixed-precision training uses less memory and is faster, especially on NVIDIA Turing-generation GPUs.
Loss scaling technique is adopted to cope with the narrow representation range of fp16.
This could improve/degrade accuracy. -->

## Demo

To use the trained models, download files, unzip, and put them under DeepDeblur-PyTorch/experiment
* [GOPRO_L1](https://drive.google.com/file/d/1AfZhyUXEA8_UdZco9EdtpWjTBAb8BbWv/view?usp=sharing)
* [REDS_L1](https://drive.google.com/file/d/1UwFNXnGBz2rCBxhvq2gKt9Uhj5FeEsa4/view?usp=sharing)
<!-- * [GOPRO_L1_amp](GOPRO_LINK) -->
<!-- * [REDS_L1_amp](REDS_LINK) -->

```bash
python main.py --precision half --save_dir SAVE_DIR --demo true --demo_input_dir INPUT_DIR_NAME --demo_output_dir OUTPUT_DIR_NAME
# demo_output_dir is by default SAVE_DIR/results
# SAVE_DIR is relative to DeepDeblur-PyTorch/experiment
```

## Differences from the original code

The default options are different from the original paper.
* RGB range is [0, 255]
* L1 loss (without adversarial loss. Usage possible. See above examples)
* Batch size increased to 16.
* Distributed multi-gpu training is recommended.
* Mixed-precision training enabled. Accuracy not guaranteed.
