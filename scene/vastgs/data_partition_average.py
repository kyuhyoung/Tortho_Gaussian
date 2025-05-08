# Author: Peilun Kang
# Contact: kangpeilun@nefu.edu.cn
# License: Apache Licence
# Project: VastGaussian
# File: data_partition.py
# Time: 5/15/24 2:28 PM


import copy
import os
import numpy as np
from typing import NamedTuple
import pickle

from scene.dataset_readers import CameraInfo, storePly
from utils.graphics_utils import BasicPointCloud
from scene.vastgs.graham_scan import run_graham_scan
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CameraPose(NamedTuple):
    camera: CameraInfo
    pose: np.array  # [x, y, z]


class CameraPartition(NamedTuple):
    partition_id: str
    cameras: list
    point_cloud: BasicPointCloud
    ori_camera_bbox: list
    extend_camera_bbox: list
    extend_rate: float

    ori_point_bbox: list
    extend_point_bbox: list


class ProgressiveDataPartitioning:
    def __init__(self, scene_info, train_cameras, model_path, m_region=2, n_region=4, extend_rate=0.2,
                 visible_rate=0.25):
        self.partition_scene = None
        self.pcd = scene_info.point_cloud
        # print(f"self.pcd={self.pcd}")
        self.model_path = model_path
        self.partition_dir = os.path.join(model_path, "partition_point_cloud")
        self.partition_ori_dir = os.path.join(self.partition_dir, "ori")
        self.partition_extend_dir = os.path.join(self.partition_dir, "extend")
        self.partition_visible_dir = os.path.join(self.partition_dir, "visible")
        self.save_partition_data_dir = os.path.join(self.model_path, "partition_data.pkl")
        self.m_region = m_region
        self.n_region = n_region
        self.extend_rate = extend_rate
        self.visible_rate = visible_rate

        if not os.path.exists(self.partition_ori_dir): os.makedirs(self.partition_ori_dir)
        if not os.path.exists(self.partition_extend_dir): os.makedirs(self.partition_extend_dir)
        if not os.path.exists(self.partition_visible_dir): os.makedirs(
            self.partition_visible_dir)
        self.fig, self.ax = self.draw_pcd(self.pcd, train_cameras)
        self.run_DataPartition(train_cameras)

    def draw_pcd(self, pcd, train_cameras):
        x_coords = pcd.points[:, 0]
        z_coords = pcd.points[:, 2]
        fig, ax = plt.subplots()
        ax.scatter(x_coords, z_coords, c=(pcd.colors), s=1)
        ax.title.set_text('Plot of 2D Points')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Z-axis')
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, 'pcd.png'), dpi=200)
        x_coords = np.array([cam.camera_center[0].item() for cam in train_cameras])
        z_coords = np.array([cam.camera_center[2].item() for cam in train_cameras])
        ax.scatter(x_coords, z_coords, color='red', s=1)
        fig.savefig(os.path.join(self.model_path, 'camera_on_pcd.png'), dpi=200)
        return fig, ax

    def draw_partition(self, partition_list):
        for partition in partition_list:
            ori_bbox = partition.ori_camera_bbox
            extend_bbox = partition.extend_camera_bbox
            x_min, x_max, z_min, z_max = ori_bbox
            ex_x_min, ex_x_max, ex_z_min, ex_z_max = extend_bbox
            rect_ori = patches.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, linewidth=1, edgecolor='blue',
                                         facecolor='none')
            rect_ext = patches.Rectangle((ex_x_min, ex_z_min), ex_x_max - ex_x_min, ex_z_max - ex_z_min, linewidth=1,
                                         edgecolor='y', facecolor='none')
            self.ax.add_patch(rect_ori)
            self.ax.add_patch(rect_ext)
        self.fig.savefig(os.path.join(self.model_path, f'regions.png'), dpi=200)
        return

    def run_DataPartition(self, train_cameras):
        if not os.path.exists(self.save_partition_data_dir):
            partition_dict = self.Camera_position_based_region_division(train_cameras)
            bbox_with_id = self.refine_ori_bbox(partition_dict)
            partition_list = self.Position_based_data_selection(partition_dict, bbox_with_id)
            self.draw_partition(partition_list)
            self.partition_scene = self.Visibility_based_camera_selection(partition_list)
            self.save_partition_data()
        else:
            self.partition_scene = self.load_partition_data()

    def save_partition_data(self):
        with open(self.save_partition_data_dir, 'wb') as f:
            pickle.dump(self.partition_scene, f)

    def load_partition_data(self):
        with open(self.save_partition_data_dir, 'rb') as f:
            partition_scene = pickle.load(f)
        return partition_scene

    def refine_ori_bbox_average(self, partition_dict):
        bbox_with_id = {}
        for partition_idx, camera_list in partition_dict.items():
            min_x, max_x = min(camera.pose[0] for camera in camera_list), max(
                camera.pose[0] for camera in camera_list)
            min_z, max_z = min(camera.pose[2] for camera in camera_list), max(camera.pose[2] for camera in camera_list)
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            bbox_with_id[partition_idx] = ori_camera_bbox

        for m in range(1, self.m_region + 1):
            for n in range(1, self.n_region + 1):
                if n + 1 == self.n_region + 1:
                    break
                partition_idx_1 = str(m) + '_' + str(n)
                min_x_1, max_x_1, min_z_1, max_z_1 = bbox_with_id[partition_idx_1]
                partition_idx_2 = str(m) + '_' + str(n + 1)
                min_x_2, max_x_2, min_z_2, max_z_2 = bbox_with_id[partition_idx_2]
                mid_z = (max_z_1 + min_z_2) / 2
                bbox_with_id[partition_idx_1] = [min_x_1, max_x_1, min_z_1, mid_z]
                bbox_with_id[partition_idx_2] = [min_x_2, max_x_2, mid_z, max_z_2]

        for m in range(1, self.m_region + 1):
            if m + 1 == self.m_region + 1:
                break
            max_x_left = -np.inf
            min_x_right = np.inf
            for n in range(1, self.n_region + 1):
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if max_x > max_x_left: max_x_left = max_x

            for n in range(1, self.n_region + 1):
                partition_idx = str(m + 1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                if min_x < min_x_right: min_x_right = min_x

            for n in range(1, self.n_region + 1):
                partition_idx = str(m) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [min_x, mid_x, min_z, max_z]

            for n in range(1, self.n_region + 1):
                partition_idx = str(m + 1) + '_' + str(n)
                min_x, max_x, min_z, max_z = bbox_with_id[partition_idx]
                mid_x = (max_x_left + min_x_right) / 2
                bbox_with_id[partition_idx] = [mid_x, max_x, min_z, max_z]

        return bbox_with_id

    def Camera_position_based_region_division(self, train_cameras):
        m, n = self.m_region, self.n_region
        CameraPose_list = []
        camera_centers = []
        for idx, camera in enumerate(train_cameras):
            pose = np.array(camera.camera_center.cpu())
            camera_centers.append(pose)
            CameraPose_list.append(
                CameraPose(camera=camera, pose=pose))

        storePly(os.path.join(self.partition_dir, 'camera_centers.ply'), np.array(camera_centers),
                 np.zeros_like(np.array(camera_centers)))

        m_partition_dict = {}
        total_camera = len(CameraPose_list)
        num_of_camera_per_m_partition = total_camera // m
        sorted_CameraPose_by_x_list = sorted(CameraPose_list, key=lambda x: x.pose[0])

        for i in range(m):
            m_partition_dict[str(i + 1)] = sorted_CameraPose_by_x_list[
                                           i * num_of_camera_per_m_partition:(i + 1) * num_of_camera_per_m_partition]
        if total_camera % m != 0:
            m_partition_dict[str(m)].extend(sorted_CameraPose_by_x_list[m * num_of_camera_per_m_partition:])

        partition_dict = {}
        for partition_idx, camera_list in m_partition_dict.items():
            partition_total_camera = len(camera_list)
            num_of_camera_per_n_partition = partition_total_camera // n
            sorted_CameraPose_by_z_list = sorted(camera_list, key=lambda x: x.pose[2])
            for i in range(n):
                partition_dict[f"{partition_idx}_{i + 1}"] = sorted_CameraPose_by_z_list[
                                                             i * num_of_camera_per_n_partition:(
                                                                                                           i + 1) * num_of_camera_per_n_partition]
            if partition_total_camera % n != 0:
                partition_dict[f"{partition_idx}_{n}"].extend(
                    sorted_CameraPose_by_z_list[n * num_of_camera_per_n_partition:])

        return partition_dict

    def extract_point_cloud(self, pcd, bbox):
        mask = (pcd.points[:, 0] >= bbox[0]) & (pcd.points[:, 0] <= bbox[1]) & (
                pcd.points[:, 2] >= bbox[2]) & (pcd.points[:, 2] <= bbox[3])
        points = pcd.points[mask]
        colors = pcd.colors[mask]
        normals = pcd.normals[mask]
        return points, colors, normals

    def get_point_range(self, points):
        x_list = points[:, 0]
        y_list = points[:, 1]
        z_list = points[:, 2]
        # print(points.shape)
        return [min(x_list), max(x_list),
                min(y_list), max(y_list),
                min(z_list), max(z_list)]

    def Position_based_data_selection(self, partition_dict, refined_ori_bbox):
        pcd = self.pcd
        partition_list = []
        point_num = 0
        point_extend_num = 0
        for partition_idx, camera_list in partition_dict.items():
            min_x, max_x, min_z, max_z = refined_ori_bbox[partition_idx]
            ori_camera_bbox = [min_x, max_x, min_z, max_z]
            extend_camera_bbox = [min_x - self.extend_rate * (max_x - min_x),
                                  max_x + self.extend_rate * (max_x - min_x),
                                  min_z - self.extend_rate * (max_z - min_z),
                                  max_z + self.extend_rate * (max_z - min_z)]
            print("Partition", partition_idx, "ori_camera_bbox", ori_camera_bbox, "\textend_camera_bbox",
                  extend_camera_bbox)
            points, colors, normals = self.extract_point_cloud(pcd, ori_camera_bbox)
            points_extend, colors_extend, normals_extend = self.extract_point_cloud(pcd, extend_camera_bbox)
            partition_list.append(CameraPartition(partition_id=partition_idx, cameras=camera_list,
                                                  point_cloud=BasicPointCloud(points_extend, colors_extend,
                                                                              normals_extend),
                                                  ori_camera_bbox=ori_camera_bbox,
                                                  extend_camera_bbox=extend_camera_bbox,
                                                  extend_rate=self.extend_rate,
                                                  ori_point_bbox=self.get_point_range(points),
                                                  extend_point_bbox=self.get_point_range(points_extend),
                                                  ))

            point_num += points.shape[0]
            point_extend_num += points_extend.shape[0]
            storePly(os.path.join(self.partition_ori_dir, f"{partition_idx}.ply"), points, colors)
            storePly(os.path.join(self.partition_extend_dir, f"{partition_idx}_extend.ply"), points_extend,
                     colors_extend)

        print(f"Total ori point number: {pcd.points.shape[0]}\n", f"Total before extend point number: {point_num}\n",
              f"Total extend point number: {point_extend_num}\n")

        return partition_list

    def get_8_corner_points(self, bbox):
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        return {
            "minx_miny_minz": [x_min, y_min, z_min],  # 1
            "minx_miny_maxz": [x_min, y_min, z_max],  # 2
            "minx_maxy_minz": [x_min, y_max, z_min],  # 3
            "minx_maxy_maxz": [x_min, y_max, z_max],  # 4
            "maxx_miny_minz": [x_max, y_min, z_min],  # 5
            "maxx_miny_maxz": [x_max, y_min, z_max],  # 6
            "maxx_maxy_minz": [x_max, y_max, z_min],  # 7
            "maxx_maxy_maxz": [x_max, y_max, z_max]  # 8
        }

    def transformPoint4x4(self, p, matrix):
        if type(p) == list:
            transformed = [
                matrix[0] * p[0] + matrix[4] * p[1] + matrix[8] * p[2] + matrix[12],
                matrix[1] * p[0] + matrix[5] * p[1] + matrix[9] * p[2] + matrix[13],
                matrix[2] * p[0] + matrix[6] * p[1] + matrix[10] * p[2] + matrix[14],
                matrix[3] * p[0] + matrix[7] * p[1] + matrix[11] * p[2] + matrix[15]
            ]
        elif type(p) == np.ndarray:
            transformed = np.array([
                matrix[0] * p[:, 0] + matrix[4] * p[:, 1] + matrix[8] * p[:, 2] + matrix[12],
                matrix[1] * p[:, 0] + matrix[5] * p[:, 1] + matrix[9] * p[:, 2] + matrix[13],
                matrix[2] * p[:, 0] + matrix[6] * p[:, 1] + matrix[10] * p[:, 2] + matrix[14],
                matrix[3] * p[:, 0] + matrix[7] * p[:, 1] + matrix[11] * p[:, 2] + matrix[15]
            ])
        return transformed

    def ndc2Pix(self, v, S):
        return ((v + 1.0) * S - 1.0) * 0.5;

    def point_in_image(self, pcd, W, H, matrix):
        points = pcd.points
        colors = pcd.colors
        normals = pcd.normals

        p_hom = self.transformPoint4x4(points, matrix)
        p_w = 1.0 / (p_hom[3, :] + 0.0000001)
        p_proj = p_hom[:3, :] * p_w[np.newaxis, :]
        point_image_x = self.ndc2Pix(p_proj[0, :], W)
        mask_x = (point_image_x >= 0) & (point_image_x <= W)

        point_image_y = self.ndc2Pix(p_proj[2, :], H)  # fix xy to xz
        mask_y = (point_image_y >= 0) & (point_image_y <= H)
        mask = mask_x & mask_y

        return points[mask], colors[mask], normals[mask]

    def Visibility_based_camera_selection(self, partition_list):
        add_visible_camera_partition_list = copy.deepcopy(partition_list)
        client = 0
        for idx, partition_i in enumerate(partition_list):
            new_points = []
            new_colors = []
            new_normals = []

            partition_id_i = partition_i.partition_id
            partition_point_bbox = partition_i.ori_point_bbox
            partition_extend_point_bbox = partition_i.extend_point_bbox
            ori_8_corner_points = self.get_8_corner_points(partition_point_bbox)
            extent_8_corner_points = self.get_8_corner_points(partition_extend_point_bbox)
            for partition_j in partition_list:
                partition_id_j = partition_j.partition_id
                if partition_id_i == partition_id_j: continue
                print(f"Now processing partition i:{partition_id_i} and j:{partition_id_j}")

                pcd_j = partition_j.point_cloud

                for cameras_pose in partition_j.cameras:
                    camera = cameras_pose.camera
                    full_proj_transform = np.array(camera.full_proj_transform.cpu()).flatten()

                    proj_8_corner_points = {}
                    for key, point in extent_8_corner_points.items():
                        p_hom = self.transformPoint4x4(point, full_proj_transform)
                        p_w = 1.0 / (p_hom[3] + 0.0000001)
                        p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]
                        point_image = [self.ndc2Pix(p_proj[0], camera.image_width),
                                       self.ndc2Pix(p_proj[2], camera.image_height)]
                        proj_8_corner_points[key] = point_image

                    pkg = run_graham_scan(list(proj_8_corner_points.values()), camera.image_width, camera.image_height)
                    if pkg["intersection_rate"] >= self.visible_rate:
                        print(f"Partition {idx} Append Camera {camera.image_name}")

                        add_visible_camera_partition_list[idx].cameras.append(cameras_pose)

                        updated_points, updated_colors, updated_normals = self.point_in_image(pcd_j, camera.image_width,
                                                                                              camera.image_height,
                                                                                              full_proj_transform)

                        new_points.append(updated_points)
                        new_colors.append(updated_colors)
                        new_normals.append(updated_normals)

                    with open(os.path.join(self.model_path, "graham_scan"), 'a') as f:
                        f.write(f"intersection_area:{pkg['intersection_area']} "
                                f"image_area:{pkg['image_area']} "
                                f"intersection_rate:{pkg['intersection_rate']}\n")

            camera_centers = []
            for camera_pose in add_visible_camera_partition_list[idx].cameras:
                camera_centers.append(camera_pose.pose)

            storePly(os.path.join(self.partition_visible_dir, f'{partition_id_i}_camera_centers.ply'),
                     np.array(camera_centers),
                     np.zeros_like(np.array(camera_centers)))

            point_cloud = add_visible_camera_partition_list[idx].point_cloud
            new_points.append(point_cloud.points)
            new_colors.append(point_cloud.colors)
            new_normals.append(point_cloud.normals)
            new_points = np.concatenate(new_points, axis=0)
            new_colors = np.concatenate(new_colors, axis=0)
            new_normals = np.concatenate(new_normals, axis=0)

            new_points, mask = np.unique(new_points, return_index=True, axis=0)
            new_colors = new_colors[mask]
            new_normals = new_normals[mask]

            add_visible_camera_partition_list[idx] = add_visible_camera_partition_list[idx]._replace(
                point_cloud=BasicPointCloud(points=new_points, colors=new_colors,
                                            normals=new_normals))
            # store_path = os.path.join(self.partition_visible_dir, str(client))
            # if not os.path.exists(store_path): os.makedirs(store_path)
            # storePly(os.path.join(self.partition_visible_dir, str(client), f"visible.ply"), new_points, new_colors)
            # client += 1
            storePly(os.path.join(self.partition_visible_dir, f"{partition_id_i}_visible.ply"), new_points,
                     new_colors)

        return add_visible_camera_partition_list

    # def format_data(self):
    #     format_data = []
    #     for partition in self.partition_scene:
    #         partition_id = partition.partition_id
    #         point_cloud = partition.point_cloud
    #         cameras = [CameraPose.camera for CameraPose in partition.cameras]
    #         cameras_extent = getNerfppNorm_partition(cameras)["radius"]
    #         format_data.append([cameras, point_cloud, cameras_extent])
    #
    #     return format_data
