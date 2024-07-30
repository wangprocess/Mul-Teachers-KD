# Multiple teachers are beneficial lightweight and noise-resistant student models for point-of-care imaging classification

## Introduction

![](./picture/image1.png)

## Data Preparation
Download the dataset and store it in the File Organization 

Run the files for data processing in the order written in the File Organization

### Dataset
[ISIC 2018](https://challenge.isic-archive.com/data/#2018) | [BUSI](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset) | [Dermnet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) 

### File Organization

``` 
├── code
    ├── dataloaders
        ├── BUSI_OverSampling      "BUSI second step"
        ├── BUSI_processing        "BUSI first step"
        ├── datasets               "pytorch dataloader"
        ├── Dermnet_OverSampling   "Dermnet second step"
        ├── Dermnet_processing     "Dermnet first step"
        ├── HairAugmentation       "ISIC2018 third step"
        ├── ISIC2018_OverSampling  "ISIC2018 second step"
        ├── ISIC2018_processing    "ISIC2018 first step"
        ├── merge_file_generate    "All Dataset merge to generate csv file"
        ├── resize                 "resize the hair images "
        └── utils                  "cal the std and mean of dataset,used for normalization"
    ├── networks                   
        ├── net_factory            "the factory of models"
        ├── ShiftMLP_base          "student model ---base"
        ├── ShiftMLP_small         "student model ---small"
        ├── SKAttention            "the SK Attention"
        ├── SwinTransformers       "the SwinTransformers"
        ├── UASwinTv2b             "the teacher model ---base"
        ├── UASwinTv2s             "the teacher model ---small"
        └── UASwinTv2t             "the teacher model ---tiny"
    ├── utils
        ├── losses
        ├── meterics
        ├── meterics2
        ├── meterics3
        ├── plots
        ├── ramps
        └── transforms
    ├── cal_parameter
    ├── test_BUSI                   "test the model in BUSI"
    ├── test_Dermnet                "test the model in Dermnet"
    ├── test_ISIC2018               "test the model in ISIC2018"
    ├── train_MTKD_BUSI             "train the model in BUSI"
    ├── train_MTKD_Dermnet          "train the model in Dermnet"
    ├── train_MTKD_ISIC2018         "train the model in ISIC2018"
    └── train_Teacher               "First train the Teacher model"
    
    ...
    "The UASwinT is the teacher model"
    "The name with KD means:use Knowledge distillation"
    "The name with MT means:use Mean Teacher model"
        
├── data [Your dataset path]
    ├── ISIC2018
        ├── ISIC2018_Task3_Test_GroundTruth
        ...
    ├── BUSI
        ├── benign
        ...
    ├── Dermnet
        ├── test
        ...
    ├── Hairs
        ...
```

## Training and Testing

First Train the Teacher Model and Select The best model.
```
python -W ignore train_Teacher.py 
```

Then Train model with Global Teacher and Assistant Teacher

```
python -W ignore train_MTKD_ISIC2018.py 
```

Finally, Evaluation model
```
python -W ignore test_ISIC2018.py 
```
