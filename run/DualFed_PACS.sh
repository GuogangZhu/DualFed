#!/bin/bash
cd ../

data_dir=$1

python main.py --dataset='pacs' --data_dir=${data_dir} --con_lambda=40.0 --con_temp=0.2