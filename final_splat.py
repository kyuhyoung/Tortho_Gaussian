import subprocess

command = [
    "python", "ortho_splat.py",
    "-s", r"./data/phantom3-ieu/",
    "--exp_name", r"phantom3-ieu",
    # "--eval",
    "--manhattan",
    "--resolution", "1",
    "--platform", "tj",
    "--pos", "0.000 0.000 0.000",
    "--rot", "90.000 0.000 0.000",
    "--load_iteration", "30000",
    "--angle_x", "90", 
    "--angle_y", "0",
    "--angle_z", "0", 
    "--scale", "0.8",
    "--fov_deg", "200",
    "--width", str(1200 * 8),
    "--height", str(1000 * 8),
    "--camera_idx", "-1"
]

subprocess.run(command)  
