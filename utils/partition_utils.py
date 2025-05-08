# -*- coding: utf-8 -*-
#        Data: 2024-06-21 17:01
#     Project: VastGaussian
#   File Name: partition_utils.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:
import os.path

import scene
from utils.camera_utils import cameraList_from_camInfos_partition

def data_partition(lp):
    from scene.dataset_readers import sceneLoadTypeCallbacks
    from scene.vastgs.data_partition import ProgressiveDataPartitioning

    scene_info = sceneLoadTypeCallbacks["Partition"](lp.source_path, lp.images, lp.man_trans, lp.eval, lp.llffhold)
    with open(os.path.join(lp.model_path, "train_cameras.txt"), "w") as f:
        for cam in scene_info.train_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    with open(os.path.join(lp.model_path, "test_cameras.txt"), "w") as f:
        for cam in scene_info.test_cameras:
            image_name = cam.image_name
            f.write(f"{image_name}\n")

    all_cameras = cameraList_from_camInfos_partition(scene_info.train_cameras + scene_info.test_cameras, args=lp)
    DataPartitioning = ProgressiveDataPartitioning(scene_info, all_cameras, lp.model_path,
                                                   lp.m_region, lp.n_region, lp.extend_rate, lp.visible_rate)
    partition_result = DataPartitioning.partition_scene

    client = 0
    partition_id_list = []
    for partition in partition_result:
        partition_id_list.append(partition.partition_id)
        camera_info = partition.cameras
        image_name_list = [camera_info[i].camera.image_name + '.jpg' for i in range(len(camera_info))]
        txt_file = f"{lp.model_path}/partition_point_cloud/visible/{partition.partition_id}_camera.txt"

        with open(txt_file, 'w') as file:
            for item in image_name_list:
                file.write(f"{item}\n")
        client += 1

    return client, partition_id_list


def read_camList(path):
    camList = []
    with open(path, "r") as f:
        lines = f.readlines()
        for image_name in lines:
            camList.append(image_name.replace("\n", ""))

    return camList


if __name__ == '__main__':
    read_camList(r"E:\Pycharm\3D_Reconstruct\VastGaussian\output\train_1\train_cameras.txt")