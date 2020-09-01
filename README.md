# CenterSeg

This repo uses Centernet and Conditional Convolutions for Instance Segmentation

> [**Objects as Points**](http://arxiv.org/abs/1904.07850),  
> [**CondInst: Conditional Convolutions for Instance Segmentation**](https://arxiv.org/abs/2003.05664)

## Installation

This repo supports both CPU and GPU Training and Inference.

```
git clone --recurse-submodules https://github.com/ajaichemmanam/CenterSeg.git

pip3 install -r requirements.txt
```

Compile DCN

```
cd src/lib/models/networks/DCNv2/

python3 setup.py build develop
```

Compile NMS
```
cd src/lib/external

python3 setup.py build_ext --inplace
```

## Pre-Trained Models

Pre-Release : [Google Drive](https://drive.google.com/drive/folders/1Uw0ucRLpyyHT0pGW2N0o5BcYdSfdNYyC?usp=sharing)

Download the most recent model (model_last_e101.pth), copy to exp/ctseg/coco_dla_1x/

Rename as model_last.pth

```
python3 demo.py ctseg --exp_id coco_dla_1x --keep_res --resume --demo ../data/coco/val2017
```

Note: Model is not completely trained (101 Epochs only). Will update later.

#### Training

###### For GPU
```
python3 main.py ctseg --exp_id coco_dla_1x --batch_size 10 --master_batch 5 --lr 1.25e-4 --gpus 0 --num_workers 4
```

###### FOR CPU
```
python3 main.py ctseg --exp_id coco_dla_1x --batch_size 2 --master_batch -1 --lr 1.25e-4 --gpus -1 --num_workers 4
```

#### Testing
```
python3 test.py ctseg --exp_id coco_dla_1x --keep_res --resume
```

## License

CenterSeg is released under the MIT License (refer to the LICENSE file for details).
This repo contains code borrowed from multiple sources. Please see their respective licenses.

## Credits

https://github.com/xingyizhou

https://github.com/Epiphqny

https://github.com/CaoWGG

