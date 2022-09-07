# Hierarchical Token Semantic Audio Transformer


## Introduction

The Code Repository for  "[HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection](https://arxiv.org/abs/2202.00874)", in ICASSP 2022.

In this paper, we devise a model, HTS-AT, by combining a [swin transformer](https://github.com/microsoft/Swin-Transformer) with a token-semantic module and adapt it in to **audio classification** and **sound event detection tasks**. HTS-AT is an efficient and light-weight audio transformer with a hierarchical structure and has only 30 million parameters. It achieves new state-of-the-art (SOTA) results on AudioSet and ESC-50, and equals the SOTA on Speech Command V2. It also achieves better performance in event localization than the previous CNN-based models. 

### Set the Configuration File: config.py

The script *config.py* contains all configurations you need to assign to run your code. 
Please read the introduction comments in the file and change your settings.

If you want to train/test your model on ESC-50, you need to set:
```
dataset_path = "your processed ESC-50 folder"
dataset_type = "esc-50"
loss_type = "clip_ce"
sample_rate = 32000
hop_size = 320 
classes_num = 50
```

### Model Checkpoints:

We provide the model checkpoints on three datasets (and additionally DESED dataset) in this [link](https://drive.google.com/drive/folders/1f5VYMk0uos_YnuBshgmaTVioXbs7Kmz6?usp=sharing). Feel free to download and test it.

## Citing
```
@inproceedings{htsat-ke2022,
  author = {Ke Chen and Xingjian Du and Bilei Zhu and Zejun Ma and Taylor Berg-Kirkpatrick and Shlomo Dubnov},
  title = {HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection},
  booktitle = {{ICASSP} 2022}
}
```
Our work is based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer), which is a famous image classification transformer model.
