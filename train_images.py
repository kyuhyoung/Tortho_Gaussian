import subprocess

command = (f'python train_vast.py -s data/phantom3-ieu- --exp_name phantom3-ieu-better'
           f' --eval'
           f' --llffhold 70 --resolution 1'
           f' --manhattan --platform tj --pos "0.000 0.000 0.000" --rot "90.000 0.000 0.000"'
           f' --m_region 2 --n_region 2'
           f' --iterations 30_000')
subprocess.run(command, shell=True)
