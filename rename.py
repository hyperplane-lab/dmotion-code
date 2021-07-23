from glob import glob
import os
import shutil
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--folders', default=1, type=int)
args = parser.parse_args()


if __name__ == '__main__':
    new_path = 'datagen/roboarm/'
    pics = sorted(glob('datagen/roboarm_1/*.png'))
    folders = args.folders
    if folders >= 2:
        for i in range(2, folders):
            pics.extend(sorted(glob(f'datagen/roboarm_{i}/*.png')))
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    eps, datasets = set(), []
    with open('datagen/roboarm/data.txt', 'w') as f:
        for pic in tqdm(pics, total=len(pics)):
            pic_name = pic.split('/')[-1]
            src = '/'.join(pic.split('/')[:2])
            if src not in datasets:
                src_f = open(src + '/' + "data.txt", "r")
                datasets.append(src)
            ep = int(pic_name.split('_')[0])
            ts = int(pic_name.split('_')[1].split('.')[0])
            if ep in eps:
                ep = max(eps) + 1 if ts == 1 else max(eps)
            action = int(src_f.readline().split()[1][0])
            eps.add(ep)
            shutil.copy(pic, new_path + '%.6d_%.2d.png' % (ep, ts))
            f.write('%.6d_%.2d %d\n' % (ep, ts, action))
    print(f'Processed {max(eps)} episodes')
