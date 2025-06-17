#!/bin/bash


if [ -n "$1" ]; then 
    DIR_DATA="$1" 
else
    DIR_DATA=/data/dabeeo/emiya
fi

if [ -n "$2" ]; then 
    GPU_IDX="$2" 
else
    GPU_IDX=3
fi

if [ -n "$3" ]; then 
    EXP_ID="$3" 
else
    #TODAY=$(date +%y%m%d) 
    EXP_ID=emiya
fi

if [ -n "$4" ]; then 
    DIR_EXP="$4" 
else
    DIR_EXP=emiya
fi



if [ -n "$5" ]; then 
    FACTOR="$5" 
else
    FACTOR=2
fi

if [ -n "$6" ]; then 
    DATA="$6" 
else
    DATA=samsung_dong
fi

if [ -n "$7" ]; then 
    MAX_ITER="$7" 
else
    MAX_ITER=30000
fi



PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py -s $DIR_DATA --exp_name $EXP_ID \
    --eval --llffhold 70 --resolution $FACTOR --model_path $DIR_EXP \
    --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000" \
    --m_region 1 --n_region 1 \
    --iterations $MAX_ITER



: << 'END'
#python train_vast.py -s data/phantom3-factory --exp_name phantom3-factory \
python train_vast.py -s /data/dabeeo/samsung_dong_mini_5 --exp_name phantom3-factory \
    --eval --llffhold 70 --resolution 1 \
    --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000" \
    --m_region 1 --n_region 1 \
    --iterations 30000
   
#--m_region 2 --n_region 2 \
END
