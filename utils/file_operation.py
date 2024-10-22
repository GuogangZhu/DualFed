import os

def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.umask(0)
        os.mkdir(save_dir, mode=0o777)