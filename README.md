## IFAN: Iterative Filter Adaptive Network for Single Image Defocus Deblurring<br><sub>Unofficial fork — tiles large images so they deblur at full resolution<br>(upstream: Official PyTorch Implementation of the CVPR 2021 Paper)</sub><br><sub>[Project](https://junyonglee.me/projects/IFAN) | [Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Lee_Iterative_Filter_Adaptive_Network_for_Single_Image_Defocus_Deblurring_CVPR_2021_paper.pdf) | [arXiv](https://arxiv.org/pdf/2108.13610.pdf) | [Supp](https://openaccess.thecvf.com/content/CVPR2021/supplemental/Lee_Iterative_Filter_Adaptive_CVPR_2021_supplemental.pdf) | [Poster](https://www.dropbox.com/s/8kpa61f1nnv0ato/IFAN_Poster.pdf?raw=1) | [Slide](https://www.dropbox.com/s/kpp6mxxxl5lah1n/IFAN_slides.pdf?raw=1)</sub><br><sub><sub>[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DmazbJPUlx4MF9-Z9llvddlywxlLWxsX?usp=sharing) [![License CC BY-NC](https://img.shields.io/badge/Replicate-Open_in_Replicate-blue.svg?style=flat)](https://replicate.ai/codeslake/ifan-defocus-deblur) [![License CC BY-NC](https://img.shields.io/badge/Anvil-Open_in_Anvil_(fastest,_but_may_be_offline)-blue.svg?style=flat)](https://YJ5YKNVB7BY5PN7Y.anvil.app/KNK4MOE27FW3VZNDQUNHAJAY)</sub></sub>

This repo contains training and evaluation code for the following paper:

> [**Iterative Filter Adaptive Network for Single Image Defocus Deblurring**](https://junyonglee.me/projects/IFAN)<br>
> [Junyong Lee](https://junyonglee.me), [Hyeongseok Son](https://sites.google.com/site/sonhspostech/), [Jaesung Rim](https://github.com/rimchang), [Sunghyun Cho](https://www.scho.pe.kr/), and [Seungyong Lee](http://cg.postech.ac.kr/leesy/)<br>
>  POSTECH <br>
> *IEEE Computer Vision and Pattern Recognition (**CVPR**) 2021*<br>


> **Unofficial fork — modified.** This is a modified copy of
> [codeslake/IFAN](https://github.com/codeslake/IFAN) as of commit `66bf136`
> (CVPR 2021, POSTECH). It is not affiliated with, or endorsed by, the original
> authors. Upstream is licensed **AGPL-3.0** ([`LICENSE`](LICENSE)), and this fork
> remains under that same licence. The additions and modifications listed below are
> © 2026 Ryan Raz ([RRaz](https://github.com/RRaz)), released under the same
> AGPL-3.0. `LICENSE` is upstream's file, unmodified.

## Modifications in this fork

**Modified 2026-10-18.** The network, the training code and the weights are
untouched. Everything added is about running large images locally, which upstream
cannot do — it downscales any input over 1920 px, so big images come back *smaller
than they went in*.

- **Never downscales.** Full resolution is the default, at any size.
- **Tiles automatically.** Anything larger than 1024 px is split into overlapping
  tiles and feathered back together, so peak GPU memory stays near 1.9 GB regardless
  of input size, instead of the ~1.8 GB *per megapixel* a whole-image pass needs
  (a 22 Mpx image needs a 19.6 GiB allocation and fails outright on a 12 GB card).
- **Stops at a ceiling rather than shrinking.** An image past the memory the machine
  has free is refused with a message naming the limit and what the image would have
  wanted, instead of quietly coming back smaller. The ceiling is worked out per
  machine, not fixed at a number that suits one.
- **Batches folders of images.** The checkpoint loads once and every image in a tree
  is processed; a per-image failure is reported without stopping the run.
- **Results stay with the sources,** in a `deblurred/` subfolder and with a `_db`
  suffix, so they cannot be mixed up with the originals.
- **Windows drop-zone runner** (`run_infer.bat`): drop a folder onto it, or put
  images in `input\` and double-click.
- **No `cog`/Replicate runtime required** — upstream `predict.py` needs it.

Start with **[`LOCAL_SETUP.md`](LOCAL_SETUP.md)** for setup, usage and verification.

For what it does on real photographs — and where it stops helping — see
**[`NOTES_FOR_PHOTOGRAPHERS.md`](NOTES_FOR_PHOTOGRAPHERS.md)**.

### Files changed from upstream

| File | Change |
|---|---|
| `predict.py` | `max_side` raised `1920` → `5760` (only affects the Cog/Replicate path) |
| `README.md` | this notice |
| `local_infer.py` | **new** — standalone tiled batch runner; 8 or 16 bits per channel, following each file |
| `run_infer.bat` | **new** — Windows drop-zone wrapper |
| `LOCAL_SETUP.md` | **new** — setup, usage, verification |
| `.gitignore` | **new** — excludes `.venv/`, `ckpt/` and generated output |
| `input/README.txt` | **new** — drop-zone instructions |
| `demo/input/input_large.jpg` | **new** — 20.7 Mpx sample that exercises tiling |
| `AFFINITY_INTEGRATION.md` | **new** — **add-on, not part of the current work**: open notes on driving this from Affinity 3 |
| `INTERFACE.md` | **new** — **add-on**: the contract the deblur and depth tools both satisfy, so a handler can drive either without touching its code |
| `NOTES_FOR_PHOTOGRAPHERS.md`, `assets/notes/` | **new** — the author's photographic notes, their figures, and the typeset PDF |

Every other upstream file is unmodified, and the full upstream history is preserved.
The pretrained weights are **not** redistributed here; download them from the links
in §3 below, as upstream intends.

### Licence and assets

`LICENSE` (GNU AGPL-3.0) covers the software in this repository — upstream's and the
additions alike — and the prose in these documents. Photographs are licensed separately,
because a software licence does not settle what an image may be used for:

- `demo/input/input.jpg` is upstream's sample image, kept exactly as upstream ships it.
- `demo/input/input_large.jpg` and the figures in `assets/notes/` are the fork author's
  own photographs. They are © 2026 Ryan Raz and licensed
  **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**: you may share
  them unchanged, with credit, for non-commercial purposes, and not as the basis of an
  altered image you distribute. **They are not covered by the AGPL-3.0 grant.** The EXIF
  on `input_large.jpg` still reads `Copyright: all rights reserved`, written before this
  licence was chosen; this notice is the current grant. If you want a sample you can
  modify and redistribute, use your own file (`demo/input/` is read by the runner like
  any other folder) or upstream's `input.jpg`.

The weights are not here, so their terms are whatever the publisher you download them
from attaches — see §3 for the links.


<p align="left">
  <a href="https://junyonglee.me/projects/IFAN">
    <img width=85% src="./assets/teaser.gif"/>
  </a><br>
</p>

## Getting Started
### Prerequisites

*Tested environment*

![Ubuntu](https://img.shields.io/badge/Ubuntu-16.0.4%20&%2018.0.4-blue.svg?style=plastic)
![Python](https://img.shields.io/badge/Python-3.8.8-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1%20&%201.8.0%20&%201.9.0%20&%201.10.2%20&%201.11.0-green.svg?style=plastic)
![CUDA](https://img.shields.io/badge/CUDA-10.2%20&%2011.1%20&%2011.3-green.svg?style=plastic)

#### 1. Environment setup
* Option 1. install from scratch
    ```bash
    $ git clone https://github.com/codeslake/IFAN.git
    $ cd IFAN

    ## for CUDA10.2
    $ conda create -y --name IFAN python=3.6 && conda activate IFAN
    $ sh install_CUDA10.2.sh

    ## for CUDA11.1 or CUDA11.3
    $ conda create -y --name IFAN python=3.8 && conda activate IFAN
    # CUDA11.1
    $ sh install_CUDA11.1.sh

    # CUDA11.3 (for amp)
    $ sh install_CUDA11.3.sh
    ```

* Option 2. docker
    ```bash
    $ nvidia-docker run --privileged --gpus=all -it --name IFAN --rm codeslake/IFAN:CVPR2021 /bin/zsh
    $ git clone https://github.com/codeslake/IFAN.git
    $ cd IFAN

    # for CUDA10.2
    $ coda activate IFAN_CUDA10.2

    # for CUDA11.1
    $ coda activate IFAN_CUDA11.1

    # for CUDA11.3 (for amp)
    $ coda activate IFAN_CUDA11.3
    ```


#### 2. Datasets
Download and unzip datasets under `[DATASET_ROOT]`:
* DPDD dataset: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%2123628&authkey=!AJTXeSG1QzF1q5A) | [Dropbox](https://www.dropbox.com/s/w9urn5m4mzllrwu/DPDD.zip?dl=1)
* PixelDP test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21487&authkey=!AAnKOuOL7yHLM94) | [Dropbox](https://www.dropbox.com/s/pw7w2bpp7pc410n/PixelDP.zip?dl=1)
* CUHK test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21486&authkey=!ABnNaUEHo63ZPKI) | [Dropbox](https://www.dropbox.com/s/zxjhzuxsxh4v0cv/CUHK.zip?dl=1)
* RealDOF test set: [OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21511&authkey=!AJu9MY_YydEc-aU) | [Dropbox](https://www.dropbox.com/s/arox1aixvg67fw5/RealDOF.zip?dl=1)

```
[DATASET_ROOT]
 ├── DPDD
 ├── PixelDP
 ├── CUHK
 └── RealDOF
```
> `[DATASET_ROOT]` can be modified with [`config.data_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L48-L49) in `./configs/config.py`.

#### 3. Pre-trained models
Download and unzip pretrained weights ([OneDrive](https://onedrive.live.com/download?resid=94530B7E5F49D254%21495&authkey=!AKmlSKrsKPa9cQ0) | [Dropbox](https://www.dropbox.com/s/qohhmr9p81u0syi/checkpoints.zip?dl=1)) under `./ckpt/`:

```
.
├── ...
├── ./ckpt
│   ├── IFAN.pytorch
│   ├── ...
│   └── IFAN_dual.pytorch
└── ...
```

## Testing models of CVPR 2021

***Important Note:** There is a mistake in the [IAC layer](https://github.com/codeslake/IFAN/blob/main/models/IAC.py#L36). The current code uses isotropic kernels in which the two separable kernels are exactly the same (per iteration).
However, the current IFAN still works, as defocus blur is usually in the isotropic shape, which the current IAC layer can handle with the isotropic kernels.*

*If you are to train the code from scratch, please update the [IAC layer](https://github.com/codeslake/IFAN/blob/main/models/IAC.py#L36) to use `kernel1` and `kernel2` as commented in the code. I will update the experimental results based on the corrected IAC layer.*

```shell
## Table 2 in the main paper
# Our final model used for comparison
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN --network IFAN --config config_IFAN --data DPDD --ckpt_abs_name ckpt/IFAN.pytorch --data_offset /data_offset --output_offset ./output

## Table 4 in the main paper
# Our final model with N=8
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_8 --network IFAN --config config_IFAN_8 --data DPDD --ckpt_abs_name ckpt/IFAN_8.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=26
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_26 --network IFAN --config config_IFAN_26 --data DPDD --ckpt_abs_name ckpt/IFAN_26.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=35
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_35 --network IFAN --config config_IFAN_35 --data DPDD --ckpt_abs_name ckpt/IFAN_35.pytorch --data_offset /data_offset --output_offset ./output

# Our final model with N=44
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_44 --network IFAN --config config_IFAN_44 --data DPDD --ckpt_abs_name ckpt/IFAN_44.pytorch --data_offset /data_offset --output_offset ./output

## Table 1 in the supplementary material
# Our model trained with 16 bit images
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_16bit --network IFAN --config config_IFAN_16bit --data DPDD --ckpt_abs_name ckpt/IFAN_16bit.pytorch --data_offset /data_offset --output_offset ./output

## Table 2 in the supplementary material
# Our model taking dual-pixel stereo images as an input
CUDA_VISIBLE_DEVICES=0 python run.py --mode IFAN_dual --network IFAN_dual --config config_IFAN --data DPDD --ckpt_abs_name ckpt/IFAN_dual.pytorch --data_offset /data_offset --output_offset ./output
```

> Testing results will be saved in `[LOG_ROOT]/IFAN_CVPR2021/[mode]/result/quanti_quali/[mode]_[epoch]/[data]/`.

> `[LOG_ROOT]` can be modified with [`config.log_offset`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L65) in `./configs/config.py`.

#### Options
* `--data`: The name of a dataset to evaluate. `DPDD` | `RealDOF` | `CUHK` | `PixelDP` | `random`. Default: `DPDD`
    * The folder structure can be modified in the function [`set_eval_path(..)`](https://github.com/codeslake/IFAN/blob/main/configs/config.py#L114-L139) in `./configs/config.py`.
    * `random` is for testing models with any images, which should be placed as `[DATASET_ROOT]/random/*.[jpg|png]`.

## Wiki
* [Logging](https://github.com/codeslake/IFAN/wiki/Log-Details)
* [Training and testing details](https://github.com/codeslake/IFAN/wiki/Training-&-Testing-Details).

## Contact
Open an issue for any inquiries.
You may also have contact with [junyonglee@postech.ac.kr](mailto:junyonglee@postech.ac.kr)

## License
![License CC BY-NC](https://img.shields.io/badge/license-GNU_AGPv3-green.svg?style=flat)<br>
This software is being made available under the terms in the [LICENSE](LICENSE) file.
Any exemptions to these terms require a license from the Pohang University of Science and Technology.

## Citation
If you find this code useful, please consider citing:

```
@InProceedings{Lee2021IFAN,
    author    = {Junyong Lee and Hyeongseok Son and Jaesung Rim and Sunghyun Cho and Seungyong Lee},
    title     = {Iterative Filter Adaptive Network for Single Image Defocus Deblurring},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2021}
}
```

