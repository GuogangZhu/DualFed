#!/bin/bash

cd ../

data_dir=$1

python main.py --dataset='officehome' --data_dir=${data_dir} --con_lambda=2.0 --con_temp=0.05