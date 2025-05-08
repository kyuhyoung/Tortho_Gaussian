import subprocess

model_path = r'\output'

command = f'SIBR_gaussianViewer_app.exe -m {model_path}'
run_path = 'viewers/bin'
subprocess.run(command, shell=True, cwd=run_path)
