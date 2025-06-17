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
    FACTOR="$4" 
else
    FACTOR=2
fi



if [ -n "$5" ]; then 
    DATA="$5" 
else
    DATA=samsung_dong
fi

if [ -n "$6" ]; then 
    MAX_ITER="$6" 
else
    MAX_ITER=30000
fi


PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$GPU_IDX python ortho_splat.py \
    --source_path $DIR_DATA --exp_name $EXP_ID \
    --manhattan \
    --resolution -1 \
    --platform tj \
    --pos "0.000 0.000 0.000" \
    --rot "90.000 0.000 0.000" \
    --load_iteration -1 \
    --angle_x 90 \
    --angle_y 0 \
    --angle_z 0 \
    --scale 0.8 \
    --fov_deg 200 \
    --width 9600 \
    --height 8000 \
    --camera_idx -1
