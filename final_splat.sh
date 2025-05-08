#!/bin/bash

# python ortho_splat.py \
#   -s "" \
#   --exp_name "" \
#   --eval \
#   --manhattan \
#   --resolution 1 \
#   --platform tj \
#   --pos "0.000 0.000 0.000" \
#   --rot "90.000 0.000 0.000" \
#   --load_iteration 7000 \
#   --angle_x 90 \
#   --angle_y 0 \
#   --angle_z 0 \
#   --scale 0.2 \
#   --width 3600 \
#   --height 2000 \
#   --camera_idx 200

python ortho_splat.py \
    -s ./data/phantom3-ieu/ \
    --exp_name phantom3-ieu\
    --manhattan \
    --resolution 1 \
    --platform tj \
    --pos "0.000 0.000 0.000" \
    --rot "90.000 0.000 0.000" \
    --load_iteration 30000 \
    --angle_x 90 \
    --angle_y 0 \
    --angle_z 0 \
    --scale 0.8 \
    --fov_deg 200 \
    --width 9600 \
    --height 8000 \
    --camera_idx -1
