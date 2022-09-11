# KE-RCNN

Official implementation of [**KE-RCNN**](https://arxiv.org/pdf/2206.10146.pdf) for part-level attribute parsing. It based on [mmdetection](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation).

## Installation
- pytorch 1.10.0 
- python 3.7.0
- [mmdet 2.25.1](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)
- [fashionpeida-API](https://github.com/KMnP/fashionpedia-api)
- einops

## Dataset
You need to download the datasets and annotations follwing this repo's formate

- [FashionPedia](https://github.com/cvdfoundation/fashionpedia)
- [knowledge_matrix](https://drive.google.com/file/d/1m1ycDqK6wvdvlLwz7jyyAuIGjyhdggBe/view?usp=sharing)

Make sure to put the files as the following structure:

```
  ├─data
  │  ├─fashionpedia
  │  │  ├─train
  │  │  ├─test
  │  │  │─instances_attribute_train2020.json
  │  │  │─instances_attribute_val2020.json
  |  |  |─train_norm_attr_knowledge_matrix.npy
  |
  ├─work_dirs
  |  ├─ke_rcnn_r50_fpn_fashion_1x
  |  |  ├─epoch32.pth
  ```

## Results and Models

### FashionPedia

|  Backbone    |  LR  | AP_iou+f1 | AP_mask_iou+f1 | DOWNLOAD |
|--------------|:----:|:---------:|:--------------:|:--------:|
|  R-50        |  1x  | 39.6      | 36.4           |[model](https://drive.google.com/file/d/1-m83sJcu9fsRNE4pNTBLmkOB8cKhPyCK/view?usp=sharing)|
|  R-101       |  1x  | 39.9      | 36.0           |[model](https://drive.google.com/file/d/1Zqa7ziBKUe3-t419dsLq6ihtYUfFLHhr/view?usp=sharing)|
|  HRNet-w18   |  1x  | 36.4      | -/-            |[model]()  |
|  Swin-tiny   |  1x  | 43.7      | 40.5           |[model](https://drive.google.com/file/d/1Y_yVRp7G6E07Mty8TIEWJe7a4dQXl44E/view?usp=sharing)|

- This is a reimplementation. Thus, the numbers are slightly different from our original paper.
## Evaluation
```
# inference
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/ke_rcnn/ke_rcnn_r50_fpn_fashion_1x.py work_dirs/ke_rcnn_r50_fpn_fashion_1x/epoch32.pth 8 --format-only --eval-options "jsonfile_prefix=work_dirs/ke_rcnn_r50_fpn_fashion_1x/ke_rcnn_r50_fpn_fashion_1x_val_result"

# eval, noted that should change the json path produce by previous step.
python eval/fashion_eval.py
```

## Training
```
# training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh configs/ke_rcnn/ke_rcnn_r50_fpn_fashion_1x.py 8
```