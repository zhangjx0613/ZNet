# ZNet
This repo holds the code of ZNet: Hybrid Transformer-CNN Architecture for Enhanced Underwater Image Semantic Segmentation

## Requirements
* Pytorch>=1.6.0, <1.9.0 (>=1.1.0 should work but not tested)
* timm==0.3.2


## Experiments

### SUIM Dataset
GPUs of memory>=4G shall be sufficient for this experiment. 

1. Preparing necessary data:
	+ downloading SUIM training and testing data from the [official site](https://irvlab.cs.umn.edu/resources/suim-dataset), put the unzipped data in `./data`.
	+ run `process.py` to preprocess all the data, which generates `data_{train, test}.npy` and `mask_{train, test}.npy`.

2. Testing:
	+ run `test_uwi.py`.

3. Training:
	+ downloading DeiT-small from [DeiT repo](https://github.com/facebookresearch/deit) to `./pretrained`.
	+ downloading resnet-34 from [timm Pytorch](https://download.pytorch.org/models/resnet34-333f7ec4.pth) to `./pretrained`.
	+ run `train_uwi.py`; you may also want to change the default saving path or other hparams as well.


Code of other tasks will be comming soon.


## Reference
Some of the codes in this repo are borrowed from:
* [Facebook DeiT](https://github.com/facebookresearch/deit)
* [timm repo](https://github.com/rwightman/pytorch-image-models)
* [PraNet repo](https://github.com/DengPingFan/PraNet)
* [Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)
* [TransFuse](https://github.com/Rayicer/TransFuse)



## Questions
Please drop an email to sunyujuan@ldu.edu.cn

