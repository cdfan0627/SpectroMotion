# SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes

### [Project page](https://cdfan0627.github.io/spectromotion/) | [Paper](https://arxiv.org/abs/2410.17249)

> **SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes**<br>
> Cheng-De Fan, 
Chen-Wei Chang, 
Yi-Ruei Liu, 
Jie-Ying Lee, 
Jiun-Long Huang, 
Yu-Chee Tseng, 
Yu-Lun Liu <br>
in CVPR 2025 <br>



## Dataset

In our paper, we use [NeRF-DS](https://jokeryan.github.io/projects/nerf-ds/) dataset and [HyperNeRF](https://hypernerf.github.io/) dataset.

We organize the datasets as follows:

```shell
├── data
│   | NeRF-DS
│     ├── as
│     ├── basin
│     ├── ...
│   | HyperNeRF
│     ├── vrig
```




## Run

### Environment

```shell
conda create -n spectromotion python=3.8
conda activate spectromotion

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt

cd nvdiffrast
pip install -e .
```



### Train


**NeRF-DS/HyperNeRF**

Because the camera poses in the HyperNeRF dataset are not very accurate, this may cause excessive growth in the number of 3D Gaussians, leading to OOM on RTX 4090. Therefore, it is recommended to train on GPUs with 48GB or more memory for the HyperNeRF dataset.

```shell
# e.g. python train.py -s data/NeRF-DS/as_novel_view -m output/NeRF-DS/as_novel_view --eval
python train.py -s path/to/your/real-world/dataset -m output/exp-name --eval
```

For the vrig-peel-banana scene from the HyperNeRF dataset, because the number of Gaussians grows excessively and causes OOM, we have modified some hyperparameters. Please use the following command for training.

```shell
python train_banana.py -s path/to/your/banana/data -m output/exp-name --eval
```



### Render
Render all the train and test images.
```shell
# e.g. python render.py -m output/NeRF-DS/as_novel_view --iteration 40000
python render.py -m output/exp-name --iteration iteration
```

For the vrig-peel-banana scene.
```shell
python render_banana.py -m output/exp-name --iteration iteration
```

### Evaluation
Evaluate all test images.
```shell
# e.g. python metrics.py --model_path "nerfds_dataset/as_novel_view/"  
python metrics.py -m output/exp-name
```
If you want to skip train and render and go directly to evaluation, please follow the command below.
```shell
gdown https://drive.google.com/uc?id=1b4dEolV9WC8Fa2ALXqwCRcZW0nTqgrEi
unzip eval_results.zip
cd eval_results
bash eval.sh
```




## BibTex

```
@article{fan2024spectromotion,
    title={SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes},
    author={Cheng-De Fan and Chen-Wei Chang and Yi-Ruei Liu and Jie-Ying Lee and 
            Jiun-Long Huang and Yu-Chee Tseng and Yu-Lun Liu},
    journal={arXiv},
    year={2024}
	}
```

And thanks to the authors of [Deformable 3D Gaussians](https://github.com/ingra14m/Deformable-3D-Gaussians) and [
GaussianShader](https://github.com/Asparagus15/GaussianShader) for their excellent code, please consider also cite their repository:

```
@article{yang2023deformable3dgs,
    title={Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction},
    author={Yang, Ziyi and Gao, Xinyu and Zhou, Wen and Jiao, Shaohui and Zhang, Yuqing and Jin, Xiaogang},
    journal={arXiv preprint arXiv:2309.13101},
    year={2023}
}

@article{jiang2023gaussianshader,
  title={GaussianShader: 3D Gaussian Splatting with Shading Functions for Reflective Surfaces},
  author={Jiang, Yingwenqi and Tu, Jiadong and Liu, Yuan and Gao, Xifeng and Long, Xiaoxiao and Wang, Wenping and Ma, Yuexin},
  journal={arXiv preprint arXiv:2311.17977},
  year={2023}
}
```
## Acknowledgments

This research was funded by the National Science and Technology Council, Taiwan, under Grants NSTC 112-2222-E-A49-004-MY2 and 113-2628-EA49-023-. The authors are grateful to Google, NVIDIA, and MediaTek Inc. for their generous donations. Yu-Lun Liu acknowledges the Yushan Young Fellow Program by the MOE in Taiwan.

