# CenterSeg

This repo uses Centernet and Conditional Convolutions for Instance Segmentation

> [**Objects as Points**](http://arxiv.org/abs/1904.07850),  
> [**CondInst: Conditional Convolutions for Instance Segmentation**](https://arxiv.org/abs/2003.05664)

## Result

These results are taken for CenterSeg model trained for 101 epochs

| type | AP    | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>s</sub> | AP<sub>m</sub> | AP<sub>l</sub> |
| ---- | ----- | --------------- | --------------- | -------------- | -------------- | -------------- |
| box  | 0.278 | 0.430           | 0.297           | 0.129          | 0.305          | 0.382          |
| mask | 0.226 | 0.387           | 0.227           | 0.078          | 0.253          | 0.340          |

| type | AR    | AR<sub>50</sub> | AR<sub>75</sub> | AR<sub>s</sub> | AR<sub>m</sub> | AR<sub>l</sub> |
| ---- | ----- | --------------- | --------------- | -------------- | -------------- | -------------- |
| box  | 0.275 | 0.455           | 0.480           | 0.265          | 0.510          | 0.674          |
| mask | 0.235 | 0.369           | 0.385           | 0.170          | 0.418          | 0.585          |

CenterPoseSeg model not trained yet

## Installation

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

Model Not yet trained. Will Update once done.

#### Training

This repo supports both CPU and GPU Training and Inference.

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
