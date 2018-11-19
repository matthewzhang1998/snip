import os
import os.path as osp

def get_dir(base_path):
    path = base_path
    ix = -1
    while osp.exists(path):
        ix += 1
        path = base_path + str(ix)

    os.makedirs(path)
    return path