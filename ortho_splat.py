import os
import sys
import math
import numpy as np
import torch
from argparse import ArgumentParser
from os import makedirs
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt

from scene import Scene, Scene_Eval
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from utils.manhattan_utils import get_man_trans
from gaussian_renderer import render, GaussianModel
from dummy_camera import DummyCamera, DummyPipeline


def generate_unique_filename(base_path, prefix):
    idx = 0
    while True:
        filename = f"{prefix}_{idx:05d}.png"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return filepath
        idx += 1


def calculate_fov(output_width, output_height, focal_x, focal_y, aspect_ratio=1.0, invert_y=False):
    fovx = 2 * math.atan((output_width / (2 * focal_x)))
    fovy = 2 * math.atan((output_height / aspect_ratio) / (2 * focal_y))
    return math.degrees(fovx), -math.degrees(fovy) if invert_y else math.degrees(fovy)


def extract_cameras_to_txt(cameras, output_path):
    camera_info_list = [str(camera) for camera in cameras]
    with open(output_path, "w") as f:
        f.write("\n".join(camera_info_list))


# def get_closest_camera_pose(camera_file, visualize=True, camera_idx=-1):
#         positions = []
#         rotations = []
#         with open(camera_file, 'r') as f:
#             lines = f.readlines()

#         camera_data = []
#         for line in lines:
#             if line.startswith("Camera"):
#                 if camera_data:
#                     parts = ''.join(camera_data).split("position=")[1].split(", rotation=")
#                     positions.append([float(x) for x in parts[0].strip('[]').split()])
#                     rotation_str = parts[1].split(']],')[0].replace('[', '').replace(']', '').split('], [')
#                     rotations.append([list(map(float, row.split())) for row in rotation_str])
#                     camera_data = []
#                 camera_data.append(line)
#             elif camera_data:
#                 camera_data.append(line)

#         if camera_data:
#             parts = ''.join(camera_data).split("position=")[1].split(", rotation=")
#             positions.append([float(x) for x in parts[0].strip('[]').split()])
#             rotation_str = parts[1].split(']],')[0].replace('[', '').replace(']', '').split('], [')
#             rotations.append([list(map(float, row.split())) for row in rotation_str])
            
#         positions = np.array(positions)

#         if camera_idx == -1:
#             center = np.mean(positions, axis=0)
#             distances = np.linalg.norm(positions - center, axis=1)
#             idx = np.argmin(distances)
#         else:
#             idx = camera_idx

#         if visualize:
#             fig = plt.figure()
#             ax = fig.add_subplot(111, projection='3d')
#             ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='Cameras')
#             ax.scatter(center[0], center[1], center[2], c='r', s=100, label='Center')
#             ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], c='g', s=100, marker='^', label='Closest')
#             ax.set_xlabel("X")
#             ax.set_ylabel("Y")
#             ax.set_zlabel("Z")
#             ax.legend()
#             plt.savefig("camera_positions.png", dpi=300)
#             plt.close()

#     return np.array(rotations[idx]).reshape(3, 3), torch.tensor(positions[idx])

def get_closest_camera_pose(camera_file, visualize=True, camera_idx=-1):
    positions = []
    rotations = []

    with open(camera_file, 'r') as f:
        lines = f.readlines()

    camera_data = []
    for line in lines:
        if line.startswith("Camera"):
            if camera_data:
                parts = ''.join(camera_data).split("position=")[1].split(", rotation=")
                positions.append([float(x) for x in parts[0].strip('[]').split()])
                rotation_str = parts[1].split(']],')[0].replace('[', '').replace(']', '').split('], [')
                rotations.append([list(map(float, row.split())) for row in rotation_str])
                camera_data = []
            camera_data.append(line)
        elif camera_data:
            camera_data.append(line)

    if camera_data:
        parts = ''.join(camera_data).split("position=")[1].split(", rotation=")
        positions.append([float(x) for x in parts[0].strip('[]').split()])
        rotation_str = parts[1].split(']],')[0].replace('[', '').replace(']', '').split('], [')
        rotations.append([list(map(float, row.split())) for row in rotation_str])

    positions = np.array(positions)

    if camera_idx == -1:
        center = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center, axis=1)
        idx = np.argmin(distances)
    else:
        idx = camera_idx

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b', label='Cameras')

        if camera_idx == -1:  # only compute center when it exists
            ax.scatter(center[0], center[1], center[2], c='r', s=100, label='Center')

        ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], c='g', s=100, marker='^', label='Selected')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.savefig("camera_positions.png", dpi=300)
        plt.close()

    return np.array(rotations[idx]).reshape(3, 3), torch.tensor(positions[idx]).float()

def render_single_camera_view(model_path, iteration, gaussians, background,
                              angle_x=0, angle_y=0, angle_z=0,
                              scale=0.2, width=1600, height=1000,
                              camera_idx=-1, fov_deg=1000.0):
    render_dir = os.path.join(model_path, "custom_view", f"ours_{iteration}", "renders")
    makedirs(render_dir, exist_ok=True)

    R, T = get_closest_camera_pose("train_cameras_output.txt", camera_idx=camera_idx)

    angle_x, angle_y, angle_z = np.deg2rad(angle_x), np.deg2rad(angle_y), np.deg2rad(angle_z)

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    R_y = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    R_z = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    R_offset = R_z @ R_y @ R_x
    R = R_offset @ R
    # R = R_z @ R_y @ R_x @ R

    S = np.diag([scale, scale, scale])
    R = S @ R
    # T = scale * T
    
    aspect_ratio = width / height
    # FoVx = 10.0
    # FoVy = 2 * math.atan((1 / aspect_ratio) * math.tan(FoVx / 2))
    FoVx_deg = fov_deg
    FoVx = math.radians(FoVx_deg) 
    FoVy = 2 * math.atan((1 / aspect_ratio) * math.tan(FoVx / 2))

    mycam = DummyCamera(R, T, FoVx, FoVy, width, height)
    pipeline = DummyPipeline()

    print("Rendering started...")
    result = render(mycam, gaussians, pipeline, background)["render"]
    print("Rendering done. Saving image...")

    path = generate_unique_filename(render_dir, "custom_view")
    torchvision.utils.save_image(result, path)
    print(f"Image saved to {path}")

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    makedirs(render_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc=f"Rendering {name} set")):
        result = render(view, gaussians, pipeline, background)["render"]
        path = generate_unique_filename(render_path, name)
        torchvision.utils.save_image(result, path)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, args=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        
        pc_dir = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}")
        pc1_path = os.path.join(pc_dir, "point_cloud_1.ply")
        pc_path = os.path.join(pc_dir, "point_cloud.ply")
 
        print(f"point_cloud_1.ply path: {pc1_path}")
        print(f"point_cloud.ply path: {pc_path}")
        
        if os.path.exists(pc1_path) and not os.path.exists(pc_path):
            print("point_cloud_1.ply to point_cloud.ply")
            os.makedirs(pc_dir, exist_ok=True)
            with open(pc1_path, "rb") as f_src, open(pc_path, "wb") as f_dst:
                f_dst.write(f_src.read())
        elif os.path.exists(pc_path) and not os.path.exists(pc1_path):
            print("point_cloud.ply to point_cloud_1.ply")
            os.makedirs(pc_dir, exist_ok=True)
            with open(pc_path, "rb") as f_src, open(pc1_path, "wb") as f_dst:
                f_dst.write(f_src.read())
                
        scene_train = Scene(dataset, gaussians, iteration, shuffle=False)
        # scene_eval = Scene_Eval(dataset, gaussians, iteration)

        background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32,
                                  device="cuda")

        extract_cameras_to_txt(scene_train.getTrainCameras(), "train_cameras_output.txt")

        render_single_camera_view(
            dataset.model_path,
            scene_train.loaded_iter,
            gaussians,
            background,
            angle_x=args.angle_x,
            angle_y=args.angle_y,
            angle_z=args.angle_z,
            scale=args.scale,
            width=args.width,
            height=args.height,
            camera_idx=args.camera_idx,
            fov_deg=args.fov_deg

        )


def main():
    parser = ArgumentParser(description="Rendering Script")
    model = ModelParams(parser, sentinel=False)
    pipeline = PipelineParams(parser)
    parser.add_argument("--load_iteration", default=-1, type=int)
    parser.add_argument("--skip_train", default=True, action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--angle_x", type=float, default=90, help="X-axis rotation in degrees")
    parser.add_argument("--angle_y", type=float, default=0, help="Y-axis rotation in degrees")
    parser.add_argument("--angle_z", type=float, default=0, help="Z-axis rotation in degrees")
    parser.add_argument("--scale", type=float, default=0.2, help="Scaling factor for the camera position")
    parser.add_argument("--width", type=int, default=1875, help="Render image width")
    parser.add_argument("--height", type=int, default=1052, help="Render image height")
    parser.add_argument("--camera_idx", type=int, default=-1, help="Manually specify camera index. If -1, auto-select closest camera.")
    parser.add_argument("--fov_deg", type=float, default=70.0, help="Horizontal field of view in degrees")

    args = parser.parse_args(sys.argv[1:])

    #print(f'args : {args}');    exit()
    args.model_path = os.path.join("./output", args.exp_name)
    #print(f'args.model_path : {args.model_path}');  exit()
    args.man_trans = get_man_trans(args)
    safe_state(args.quiet)

    print("Rendering model at:", args.model_path);  #exit()
    print("Manhattan Transformation:", args.man_trans)

    render_sets(
        model.extract(args),
        args.load_iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args=args
    )


if __name__ == "__main__":
    main()
