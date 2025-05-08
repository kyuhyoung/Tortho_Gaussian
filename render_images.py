# import subprocess

# command = 'python render.py -m your/data/path'
# subprocess.run(command, shell=True)


import subprocess

# command = (f'python render_dummy.py'
#            f' -s output\phantom3-ieu7000_8r'
#            f' --exp_name output\phantom3-ieu7000_8r'
#            f' --eval'
#            f' --manhattan'
#            f' --resolution 1'
#            f' --platform tj'
#            f' --pos "0.000 0.000 0.000"'
#            f' --rot "90.000 0.000 0.000" '
#            f' -load_iteration 30_000'
#            )
command = [
    "python", "ortho_splat.py",
    "-s", r"./data/phantom3-ieu-",
    "--exp_name", r"phantom3-ieu-new_3",
    "--eval",
    "--manhattan",
    "--resolution", "1",
    "--platform", "tj",
    "--pos", "0.000 0.000 0.000",
    "--rot", "90.000 0.000 0.000",
    "--load_iteration", "30000"
]

subprocess.run(command)