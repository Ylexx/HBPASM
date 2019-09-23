# HBPASM
 A  pytorch implementation of Fine-Grained Classification via  Hierarchical Bilinear Pooling with Aggregated Slack Mask (HBPASM).


## Requirements
- python 2.7
- pytorch 0.4.1

## Train

Step 1. 
- Download the resnet-34 pre-training parameters.
[resnet34-parameters-download](https://pan.baidu.com/s/14KHkdAutkHLCdtsHu5VC7g)
or
[resnet34-parameters-download](https://download.pytorch.org/models/resnet34-333f7ec4.pth)


- Download the CUB-200-2011 dataset.
[CUB-download](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)

Step 2. 
- Set the path to the dataset and resnet parameters in the code.

Step 3. Train the fc_layer and proj-layer only.
- python train_firststep.py

Step 4. Fine-tune all layers. It gets an accuracy of around 86.8% on CUB-200-2011 when using resnet-34.
- python train_finetune.py



![image](https://github.com/Ylexx/HBPASM/blob/master/bird_mask.jpg)
Visualization of independent masks and the aggregated mask generated on three convolutional layers. The aggregated mask generates better RoIs with fewer background regions owing to the combination of the multiple mask maps.
