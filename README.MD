# S<sup>3</sup>GM: Learning spatiotemporal dynamics with a pretrained generative model

This repository is the source code for paper "[Learning spatiotemporal dynamics with a pretrained generative model](https://doi.org/10.1038/s42256-024-00938-z)" published in _Nature Machine Intelligence_ (2024).

Our proposed method (**S**parse-**S**ensor-assisted **S**core-based **G**enerative **M**odel, S<sup>3</sup>GM) tackles the challenge of reconstructing nonlinear spatiotemporal dynamics from sparse sensor data. S<sup>3</sup>GM contains two stages: __(1) learning complex dynamics through unconditioned pretraining; (2) using dynamic-informed sampling based on sparse measurements for accurate predictions__. It excels in zero-shot reconstruction, showing strong accuracy and generalization even with extremely sparse or noisy data. 

To reproduce the results presented in our paper, you can follow our instructions below.

## 1. Install Dependancies

Create and enter your own Python virtual environment, and then run

    pip install -r requirements.txt



## 2. Data Preparation

You can download the datasets for a 1D case (Kuramoto-Sivashinsky equation, KSE) and a 2D case (Kolmogorov flow) via Zenodo repository at https://doi.org/10.5281/zenodo.13925732. After downloading, copy the files to './data' directory under the root path of the project (by default, the training and test datasets for KSE are put under the './data/kse' directory, and the training and test datasets for Kolmogorov flow are under the './data/kolmogorov' directory).

For customized 1D/2D datasets, you can follow the data format/shape described in https://doi.org/10.5281/zenodo.13925732.



## 3. Pretraining Stage

To start pretraining from scratch, run the following command for KSE (with a single GPU):

    CUDA_VISIBLE_DEVICES=0 python train.py --data kse --version <version> --dims 1 --epochs 200 --batch_size 50 --train_split 0.9 --num_components 1 --num_conditions 1 --attn_resolutions 16 --ch_mult 1 2 4 8 --data_location ./data/kse/KSE_train.npy --verbose 1

or Kolmogorov flow (with a single GPU):

    CUDA_VISIBLE_DEVICES=0 python train.py --data kol --version <version> --dims 2 --epochs 200 --batch_size 32 --train_split 0.8 --num_components 2 --num_conditions 2 --attn_resolutions 8 16 --ch_mult 1 2 4 8 --data_location ./data/kolmogorov/kolmogorov_flow_train.npy --verbose 1

where '--data' identifies the used dataset, '--version' can be customized to indicate specific run, and '--data_location' is the path to your saved dataset.


## 4. Generating stage

If you do not want to waste time for pretraining, we also provided the pretrained checkpoints for generaing stage only, which can also be downloaded via Zenodo repository at https://doi.org/10.5281/zenodo.13925732. The checkpoint file (i.e., 'kse_v0' for KSE and 'kol_v0' for Kolmogorov flow) should be copied to the './results' directory.

To better analyze and visualize the results, we provide two notebooks for KSE and Kolmogorov flow generation, respectively. Please open **'sample_kse.ipynb'** or **'sample_kol.ipynb'**, and run the blocks sequentially to perform the generating stage and results analysis.
