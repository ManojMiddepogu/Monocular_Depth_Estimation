# [VA-DepthNet++]
We build our code upon [VA-DepthNet](https://github.com/cnexah/VA-DepthNet).

## Environment Setup

We recommend you to set up a conda environment with the requried packages.
```
conda env create -f environment.yml
```

## Training

Download the pretrained encoder backbone from [here](https://github.com/microsoft/Swin-Transformer). The code currently supports Swin-T, Swin-S, Swin-L backbones.

Training VA-DepthNet model for NYUv2 with Swin-T backbone
```
python vadepthnet/train.py configs/tiny_arguments_train_nyu.txt
```

Training VA-DepthNet++ model for NYUv2 with Swin-T backbone with window size 2
```
python vadepthnet/train.py configs/window_2_arguments_train_nyu.txt
```

Training VA-DepthNet++ model for NYUv2 with Swin-T backbone with window size 3
```
python vadepthnet/train.py configs/window_3_arguments_train_nyu.txt
```

Training Flow Gradient model for NYUv2 with Swin-T backbone
```
python vadepthnet/train.py configs/flow_tiny_arguments_train_nyu.txt
```

## References

1. [VA-DepthNet: A Variational Approach to Single Image Depth Prediction](https://openreview.net/forum?id=xjxUjHa_Wpa)
