from glob import glob
import shutil
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folders', default=1, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    if not os.path.exists('datagen/roboarm_mpc'):
        os.mkdir('datagen/roboarm_mpc')
    n_folders = args.folders
    states = {}
    for i in range(1, n_folders + 1):
        imgs = glob(f'datagen/roboarm_mpc{i}/*.png')
        for img_path in imgs:
            new_path = img_path.split('/')
            new_path[1] = 'roboarm_mpc'
            new_path = '/'.join(new_path)
            shutil.copy(img_path, new_path)
        state = json.load(open(f'datagen/roboarm_mpc{i}/states.json'))
        states = {**states, **state}
    json.dump(states, open('datagen/roboarm_mpc/states.json', 'w'))
