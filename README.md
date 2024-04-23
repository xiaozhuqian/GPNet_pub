# GPNet
This is the PyTorch version repo for [Boosting fish counting in sonar images with global attention and point supervision](https://www.sciencedirect.com/science/article/abs/pii/S0952197623012770)

## Data preparation
GPNet is evaluated on processed ARIS dataset [link](https://zenodo.org/records/4717411).

The file structure is：

root
    |-train
        |-images
            |-0.jpg
            ...
        |-groundtruths
            |-density_map
                |-0.h5
                ...
            |-fish_mask
                |-0.h5
                ...
            |-points
                |-0.txt
                ...
    |-val
    |-test

## Prerequisites
Install requirements.txt before training.

## Training
```python train.py --data_dir './data/arisdata' --save_dir './outputs' --max_model_num 1 --vis_env 'GPNet' --gpu '4,5' --batch_size 8 --lr 1e-5 --epochs 500```

## Testing
Use test.py to test a batch of images.
