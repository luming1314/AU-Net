<div align="center">
<h1>AU-Net: Adaptive Unified Network for </br> Joint Multi-modal Image Registration and Fusion</h1>

[**Ming Lu**](https://luming1314.github.io/),  Min Jiang, Xuefeng Tao, Jun Kong <br>

Jiangnan University

<!-- <sup>*</sup>corresponding authors -->

<a href='https://doi.org/10.1109/TIP.2025.3586507'><img src='https://img.shields.io/badge/DOI-10.1109%2FTIP.2025.3586507-blue'></a>

</div>

**AU-Net** integrates registration and fusion into a unified framework, eliminating the need for intermediate registered images.


## 📢 News
- **2025-8-10** 🔥 Code and model released!

## 📝 TODOs:
- [x] Image-to-image translation module

## ✨ Usage

### Quick start
#### 1. Clone this repo and setting up environment
```sh
git clone https://github.com/luming1314/AU-Net.git
cd AU-Net
conda create -n AU-Net python=3.8 -y
conda activate AU-Net
pip install -r requirements.txt
```

#### 2. Download pre-trained models

You can download our pre-trained models for a quick start.

| Google Drive | Baidu Netdisk | Description
| :--- | :--- | :----------
|[AU-Net](https://drive.google.com/file/d/1XL97XsMB1C5CbJ-uLHkOYalWj2QdDrHD/view?usp=sharing) |[AU-Net](https://pan.baidu.com/s/19Tc8d1yvuDJglf742XlNeg?pwd=bjuk) |Pre-trained AU-Net model

#### 3. Test

To test AU-Net, execute `test.sh`:

```shell
sh test.sh
```

## ⚙️ Training

### Prepare data
AU-Net is trained on the [NirScene](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) dataset and evaluated on both the [NirScene](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) and [RoadScene](https://github.com/hanna-xu/RoadScene) datasets. We recommend using our preprocessed dataset for training:

| Baidu Netdisk| Description
| :--- |:----------
|[Registration and Fusion](https://pan.baidu.com/s/18kdgwXHzajfmSs2faV9oDw?pwd=g31m) | Training and Testing Datasets for AU-Net
### Dataset structure
```markdown
📦 datasets
├── 📂 test
│   └── 📂 NirScene              # Dataset name
│       ├── 📂 ir                # Infrared images
│       ├── 📂 ir_warp           # Warped Infrared Images
│       └── 📂 vi                # Visible images
└── 📂 train
    └── 📂 NirScene 
        ├── 📂 src_r2v           # Infrared images
        ├── 📂 src_v2r           # Visible images
        ├── 📂 tar_r2v           # Pseudo Visible images
        └── 📂 tar_v2r           # Pseudo Infrared images
```
### Training AU-Net
To train AU-Net, execute `train.sh`:
```shell
sh train.sh
```
## 💪 I2I-DDPM
For the image-to-image translation module used to guide AU-Net training, see our [I2I-DDPM](https://github.com/luming1314/I2I-DDPM) repository.
## 👏 Acknowledgment
Our work is standing on the shoulders of giants. We want to thank the following contributors that our code is based on:
* SuperFusion: https://github.com/Linfeng-Tang/SuperFusion
* T2V-DDPM: https://github.com/Nithin-GK/T2V-DDPM
* ODConv: https://github.com/OSVAI/ODConv
## 🎓 Citation

If AU-Net is helpful to your work, please cite our paper via:

```
@ARTICLE{11079838,
  author={Lu, Ming and Jiang, Min and Tao, Xuefeng and Kong, Jun},
  journal={IEEE Transactions on Image Processing}, 
  title={AU-Net: Adaptive Unified Network for Joint Multi-Modal Image Registration and Fusion}, 
  year={2025},
  volume={34},
  number={},
  pages={4721-4735},
  doi={10.1109/TIP.2025.3586507}
  }

```