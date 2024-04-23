#!/bin/bash

python train.py --save_dir '/home/data/outputs/segLoss1.0' --vis_env 'segLoss1.0' --gpu '2,3' --seg_coe 1.0


python train.py --save_dir '/home/data/outputs/segLoss0.0' --vis_env 'segLoss0.0' --gpu '2,3' --seg_coe 0.0
