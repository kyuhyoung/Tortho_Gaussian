#!/bin/bash

python train.py -s /data/dabeeo/samsung_dong_mini_5 --exp_name phantom3-factory \
    --eval --llffhold 70 --resolution 1 \
    --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000" \
    --m_region 1 --n_region 1 \
    --iterations 30000



: << 'END'
#python train_vast.py -s data/phantom3-factory --exp_name phantom3-factory \
python train_vast.py -s /data/dabeeo/samsung_dong_mini_5 --exp_name phantom3-factory \
    --eval --llffhold 70 --resolution 1 \
    --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000" \
    --m_region 1 --n_region 1 \
    --iterations 30000
   
#--m_region 2 --n_region 2 \
END
