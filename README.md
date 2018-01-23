# FPN with PyTorch
This project is mainly based on [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch).

For details about FPN please refer to the [paper](https://arxiv.org/abs/1612.03144).
Feature Pyramid Networks for Object Detection
### Reference Project

-  [faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
-  [FPN](https://github.com/unsky/FPN)
-  [Mask_RCNN](https://github.com/matterport/Mask_RCNN)

### Training on Pascal VOC 2007

This project use ResNet-101 model converted from Caffe, and you can get it following [RuotianLuo-pytorch-ResNet](https://github.com/ruotianluo/pytorch-resnet).

Since the program loading the data in `pytorch_FPN/data` by default,
you can set the data path as following.
```bash
cd pytorch_FPN
mkdir data
cd data
ln -s $VOCdevkit VOCdevkit2007
```

Then you can set some hyper-parameters in `train.py` and training parameters in the `.yml` file.

```bash
python train.py
```

### Evaluation
Set the path of the trained model in `test.py`.
```bash
cd pytorch_FPN
python demo.py
```

![image]（https://github.com/xingmimfl/pytorch_FPN/blob/master/demo/out.jpg)

As [FPN](https://github.com/unsky/FPN) pointed out, anchor sizes for COCO may not suitable for VOC data, you can 
change anchor size to get better performance.


License: MIT license (MIT)
