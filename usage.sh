#!/usr/bin/env bash

#   do this once after docker run
pip3 install submodules/diff-gaussian-rasterization-ortho submodules/simple-knn
#pip3 install -e submodules/RoMa
#pip3 install submodules/fused-ssim 
#pip install submodules/diff-gaussian-rasterization

bash ./train_images.sh
#bash ./final_splat.sh

: << 'END'
#DATASETS="mill_19_building mill_19_rubble"
#DATASETS="mill_19_building"
#DATASETS="mill_19 urban_scene_3d"
#DATASETS="mill_19"
DATASETS="dabeeo"
#SCENES="building"
#SCENES="rubble"
#SCENES="samsung_dong_mini_21"
#SCENES="samsung_dong_mini_5"
SCENES="samsung_dong_mini_5 samsung_dong_mini_21"
FACTORS="4 2"
SH_DEGS="3 0"
GPU=0
N_ITER_MAX=300000
N_ITER_DENSIFY=300000
#N_ITER_MAX=20
#N_ITER_DENSIFY=20

for DATASET in $DATASETS; do
    for FACTOR in $FACTORS; do
        #echo "FACTOR : $FACTOR"
        for SH_DEG in $SH_DEGS; do
            for SCENE in $SCENES; do
                EXP_SUFFIX=edgs_f_${FACTOR}_sh_${SH_DEG}_dens_${N_ITER_DENSIFY}
                DIR_DATA=/data/${DATASET}/${SCENE}
                #bash ./scripts/preprocess/colmap_mapping.sh $DIR_DATA $DIR_DATA 0 ./voc_tree/vocab_tree_flickr100K_words256K.bin 100
                ##bash ./scripts/preprocess/colmap_mapping.sh /data/${SCENE} $OUT ./voc_tree/vocab_tree_flickr100K_words256K.bin 100 0
                #python -m scripts.preprocess.meganerf_to_colmap /data/${DATASET} $SCENE
                ##cd scripts/train && DATASET=mipnerf360 && ./train_nvs.sh 0 $EXP_SUFFIX $DATASET gaussian_splatting && cd -
        
                #CONFIG=${DATASET}_${SCENE} && cd scripts/train && ./train_nvs.sh 0 $EXP_SUFFIX $CONFIG gaussian_splatting && cd -
                DIR_OUT=./output/${DATASET}_${SCENE}_${EXP_SUFFIX}
                #CUDA_DEVICE_ORDER=PCI_BUS_ID \
                CUDA_VISIBLE_DEVICES=$GPU \
                python train.py \
                    train.gs_epochs=$N_ITER_MAX \
                    train.no_densify=False \
                    gs.dataset.source_path=$DIR_DATA \
                    gs.dataset.model_path=$DIR_OUT \
                    gs.dataset.resolution=$FACTOR \
                    gs.opt.densify_until_iter=$N_ITER_DENSIFY \
                    gs.sh_degree=$SH_DEG \
                    init_wC.matches_per_ref=20000 \
                    init_wC.nns_per_ref=3 \
                    init_wC.num_refs=180 \
                    wandb.entity=kyuhyoung 
            done
        done
    done
done
END
