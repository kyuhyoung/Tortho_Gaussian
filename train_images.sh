#!/bin/bash

python train_vast.py -s data/phantom3-factory --exp_name phantom3-factory \
    --eval --llffhold 70 --resolution 1 \
    --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000" \
    --m_region 2 --n_region 2 \
    --iterations 30000
