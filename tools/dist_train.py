import os
import subprocess
from os.path import dirname, join
 
# 设置环境变量和参数
config = "./projects/configs/maptr/maptr_tiny_r50_24e.py"
gpus = "1"
port = "28509"
 
# 指定Python解释器路径
python_interpreter = "/home/nio/miniconda3/envs/open-mmlab/bin/python"
 
# 获取当前脚本所在目录的父目录，并更新PYTHONPATH环境变量
current_script_dir = dirname(__file__)
parent_dir = join(current_script_dir, "..")
os.environ['PYTHONPATH'] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"
 
# 构建命令行命令
command = [
    python_interpreter, '-m', 'torch.distributed.launch',
    '--nproc_per_node=' + gpus,
    '--master_port=' + port,
    join(dirname(__file__), 'train.py'), config,
    '--launcher', 'pytorch'
]

# 执行命令
subprocess.run(command)