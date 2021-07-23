import pandas as pd
import shutil
import json


if __name__ == '__main__':
    mpc_json = {}
    meta_json = json.load(open('datagen/roboarm_mpc/states.json'))
    metadata = pd.read_csv('datagen/mpc.csv')
    for i, row in metadata.iterrows():
        ep = row.episode
        start_idx = row.start_idx
        end_idx = row.end_idx
        mpc_json[i] = {}
        for k, v in meta_json[str(ep)].items():
            print(ep, start_idx, end_idx)
            mpc_json[i][k] = [v[start_idx], v[end_idx]]
        start_img_name = 'datagen/roboarm_mpc/%.6d_%.2d.png' % (ep, start_idx + 1)
        end_img_name = 'datagen/roboarm_mpc/%.6d_%.2d.png' % (ep, end_idx + 1)
        new_init_name0 = 'datagen/roboarm_mpc/%.6d_init0.png' % (i)
        new_init_name1 = 'datagen/roboarm_mpc/%.6d_init1.png' % (i)
        new_end_name = 'datagen/roboarm_mpc/%.6d_goal.png' % (i)
        shutil.copy(start_img_name, new_init_name0)
        shutil.copy(start_img_name, new_init_name1)
        shutil.copy(end_img_name, new_end_name)
    json.dump(mpc_json, open('datagen/roboarm_mpc/state.json', 'w'))
