#!/usr/bin/env bash

#   do this once after docker run
GPU_IDX=1
DATA=samsung_dong_mini_5
FACTOR=4
MAX_ITER=30000
DIR_DATA=/data/dabeeo/${DATA} 
#ROT_DEG_X=90.000
ROT_DEG_X=0.000
ANG_DEG_X=0

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <arg1> [<arg2> ...]" >&2
    echo "Error: at least 1 arguments required, but got $#." >&2
    exit 1
else
    if pip3 show diff_gaussian_rasterization > /dev/null 2>&1; then
        echo "🔍 diff_gaussian_rasterization 이 설치되어 있습니다. 제거를 시작합니다…"
        pip3 uninstall -y diff_gaussian_rasterization && echo "✅ diff_gaussian_rasterization이 성공적으로 제거되었습니다." || { echo "❌ 제거 중 오류가 발생했습니다." >&2; exit 1; }
    else
        echo "ℹ️ diff_gaussian_rasterization 이 설치되어 있지 않습니다. 아무 작업도 하지 않습니다."
    fi
    lower="${1,,}"
    if [ "$lower" = "train" ]; then
        echo "It is training mode !!!"
        TODAY=$(date +%y%m%d) 
        EXP_ID=${DATA}_f_${FACTOR}_it_${MAX_ITER}_${TODAY}
        DIR_EXP=./output/${EXP_ID} 
        if pip3 install submodules/diff-gaussian-rasterization submodules/simple-knn; then
            echo "✅ submodule 설치 성공, 학습을 시작합니다."
            PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$GPU_IDX python train.py -s $DIR_DATA --exp_name $EXP_ID --eval --llffhold 70 --resolution $FACTOR --model_path $DIR_EXP --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "${ROT_DEG_X} 0.000 0.000" --m_region 1 --n_region 1 --iterations $MAX_ITER
        else
            echo "❌ diff-gaussian-rasterization 설치에 실패했습니다." >&2
            exit 1           
        fi
    else      
        echo "It is rendering mode !!!"
        EXP_ID=${DATA}_f_${FACTOR}_it_${MAX_ITER}_250616
        DIR_EXP=./output/${EXP_ID} 
        if pip3 install submodules/diff-gaussian-rasterization-ortho submodules/simple-knn; then
            echo "✅ submodule 설치 성공, 학습을 시작합니다."
            PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=$GPU_IDX python ortho_splat.py --source_path $DIR_DATA --exp_name $EXP_ID --manhattan --resolution $FACTOR --platform tj --pos "0.000 0.000 0.000" --rot "${ROT_DEG_X} 0.000 0.000" --load_iteration -1 --angle_x $ANG_DEG_X --angle_y 0 --angle_z 0 --scale 0.8 --fov_deg 200 --width 9600 --height 8000 --camera_idx -1
        else
            echo "❌ diff-gaussian-rasterization 설치에 실패했습니다." >&2
            exit 1
        fi      
    fi       
fi

#pip3 install -e submodules/RoMa
#pip3 install submodules/fused-ssim 
#pip install submodules/diff-gaussian-rasterization

#bash ./train_images.sh $DIR_DATA $GPU_IDX $EXP_ID $DIR_EXP $FACTOR $DATA $MAX_ITER
#bash ./final_splat.sh $DIR_DATA $GPU_IDX $EXP_ID 

