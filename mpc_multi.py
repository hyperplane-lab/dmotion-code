# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image
import json

# from models import Model

from test import frame2img
from test import oneshot_act2mats


from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.vision_sensor import VisionSensor
from tqdm import tqdm, trange
from os.path import join
import numpy as np
from random import shuffle
from PIL import Image
import math
from datagen.utils import make_or_clean_dir
import time
import cv2
from scipy.spatial.distance import cdist
import random
from itertools import product
from torchvision.utils import save_image
from config import parser

args = parser.parse_args()
if not os.path.exists(args.test_save_path + '_' + args.model_type):
    os.mkdir(args.test_save_path + '_' + args.model_type)
distance_measure = 'manhattan' if args.manhattan else ('l2' if args.l2 else 'euc')
args.test_save_path = args.test_save_path + '_' + args.model_type + '/' + str(args.num_iter) + '_' \
    + str(args.len_path) + '_' + str(args.sum_tra) + '_' + str(args.total_steps) + '_' + distance_measure


class RepObject:
    """Interactive objects in PyRep"""
    def __init__(self, object_type, size, color, object_area, position_range, name):
        self.shape = Shape.create(type=object_type,
                       size=size,
                       color=color,
                       static=False, respondable=True)
        self.shape.set_bullet_friction(0.8)
        self.position = object_area
        self.position_range = position_range
        self.name = name

    def randomize_position(self, new_object_area, random_range):
        self.position_range = random_range
        self.position = [pos + np.random.uniform(*self.position_range) if i < 2 else pos for i, pos in enumerate(new_object_area)]
        self.shape.set_position(self.position)


class VisionSensors:
    """A list of vision sensors in PyRep for capturing snapshots"""
    def __init__(self, sensor_names):
        self.sensors = {name: VisionSensor(name) for name in sensor_names}

    def capture(self, episode, timestep):
        for name, cam in self.sensors.items():
            view = name.split('_')[-1]
            img = Image.fromarray((cam.capture_rgb() * 255).astype(np.uint8))
            if 'side' in name:
                img = img.rotate(180, expand=True).resize((args.size, args.size))
            else:
                img = img.rotate(270, expand=True).resize((args.size, args.size))
            return img


def get_random_path(gripper, next_move):
    target_position = gripper.get_position()
    if next_move[0]:
        target_position[0] += next_move[0]
    elif next_move[1]:
        target_position[1] += next_move[1]
    target_position[-1] = TABLE_LEVEL
    return agent.get_linear_path(position=target_position, euler=[0, math.radians(180), 0], steps=2)


def close_gripper(gripper):
    done = False
    while not done:
        done = gripper.actuate(0., velocity=1.)
        pr.step()


DATA_PATH = 'datagen/roboarm_mpc'
SENSORS = ['Vision_sensor']#, 'Vision_sensor_front', 'Vision_sensor_side']
TABLE_LEVEL = 0.77
RANDOM_MOVES = {
    0: np.array([-0.01, 0, 0]),
    1: np.array([0.01, 0, 0]),
    2: np.array([0, 0.01, 0]),
    3: np.array([0, -0.01, 0])}
SCENE_FILE = f'{args.data_path}/%.6d_init1.ttt' % args.idx_mpc

pr = PyRep()
pr.launch(SCENE_FILE, headless=True)
pr.start()
agent = Panda()
gripper = PandaGripper()
sensors = VisionSensors(SENSORS)
policy = 'random'

objects = {
            'red_cube': Shape('Cuboid'),
            'pink_cylinder': Shape('Cylinder'),
            'green_cube': Shape('Cuboid0'),
            'blue_cylinder': Shape('Cylinder0'),
            'yellow_cube': Shape('Cuboid1'),
            'cyan_cylinder': Shape('Cylinder1'),
            }
starting_joint_positions = agent.get_joint_positions()
close_gripper(gripper)


class Roboarm:
    def __init__(self):
        self.agent = agent
        self.gripper = gripper
        self.sensors = sensors
        self.pr = pr
        self.objects = objects
        self.step_size = 0.08

    def get_positions(self):
        positions = []
        for obj in self.objects.values():
            positions.append(obj.get_position())
        return positions

    def state2positions(self, states):
        return [np.array(value) for key, value in states.items() if key != 'gripper']

    def step(self, action, moved=None):
        try:
            path = self.get_path(action)
        except:
            return None
        position_ = self.gripper.get_position()[:-1]
        done = False
        if moved is None:
            moved = np.array([0., 0.])
        while not done:
            done = path.step()
            pr.step()
            position = gripper.get_position()[:-1]
            disp = position - position_
            moved += disp
            if np.linalg.norm(abs(moved), ord=np.inf) > self.step_size:
                return
        if np.linalg.norm(abs(moved), ord=np.inf) < self.step_size:
            self.step(action, moved)

    def get_path(self, action):
        target_position = self.gripper.get_position()
        if action[0]:
            target_position[0] += action[0]
        elif action[1]:
            target_position[1] += action[1]
        target_position[-1] = TABLE_LEVEL
        return agent.get_linear_path(position=target_position, euler=[0, math.radians(180), 0], steps=2)


def clasp_act2mats(pairs, actions, model, device, args):
    act2mats = [{}]
    pth = os.path.join('datagen/demonstration', args.dataset_name)
    for p, a in zip(pairs, actions):
        frame0 = Image.open(os.path.join(pth, p[0]))
        frame0 = np.asarray(frame0.resize((args.size,args.size)), dtype=np.float32) / 127.5 - 1.0
        frame0 = torch.from_numpy(np.transpose(frame0, (2, 0, 1))).to(device).unsqueeze(0)
        frame1 = Image.open(os.path.join(pth, p[1]))
        frame1 = np.asarray(frame1.resize((args.size,args.size)), dtype=np.float32) / 127.5 - 1.0
        frame1 = torch.from_numpy(np.transpose(frame1, (2, 0, 1))).to(device).unsqueeze(0)

        out, _ = model.MotionEnc.forward(frame0, frame1)
        trans_mat = out.squeeze(0).cpu().detach().numpy()
        act2mats[0][a] = trans_mat
    return act2mats[0]


def find_pos(img, rgb, cnt_threshold, show=0, normalize=True):
    if normalize:
        img = (img.squeeze().detach().cpu().numpy() + 1) * 127.5
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'temp{random_int}.png', img)
    img = cv2.imread(f'temp{random_int}.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, rgb[0], rgb[1])
    cnt = mask.sum() / 255
    xy = np.argwhere(mask == 255)
    if cnt >= cnt_threshold and show:
        cp = np.array(img)
        name = 10000
        for i, j in xy:
            cp[i][j] = [255, 255, 255]
        Image.fromarray(cp.astype(np.uint8)).save(f'datagen/testing/{name}.png')
    if cnt < cnt_threshold:
        return -1., -1.
    x_ = xy.mean(0)[0]
    y_ = xy.mean(0)[1]
    return x_, y_


def get_distances(pred, goal, cnt_threshold=10, normalize=True):
    poserr = []

    color_dict = {'green': [(40, 50, 50), (70, 255, 255)],
                  'yellow': [(25, 65, 65), (36, 255, 255)],
                  'blue': [(110, 150, 150), (125, 255, 255)],
                  'cyan': [(80, 150, 150), (95, 255, 255)],
                  'red': [(0, 100, 100), (11, 255, 255)],
                  'pink': [(138, 75, 75), (162, 255, 255)]
                }

    for c in color_dict.values():
        x1, y1 = find_pos(goal, np.array(c), cnt_threshold, normalize=normalize)
        if x1 < 0 or y1 < 0:
            continue
        x2, y2 = find_pos(pred, np.array(c), cnt_threshold, normalize=normalize)
        if x2 < 0 or y2 < 0:
            if args.hide_penalty:
                pos_eror = args.hide_penalty
            else:
                continue
        if args.manhattan:
            pos_eror = cdist(np.array([[x1, y1]]), np.array([[x2, y2]]), metric='cityblock')
        else:
            pos_eror = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        poserr.append(pos_eror)
    return np.mean(poserr)


def compute_avg_distance(now, goal, GT=False):
    if GT:
        tot_dis = 0.0
        n_objs = len(now)
        for now_pos, goal_pos in zip(now, goal):
            tot_dis += np.linalg.norm(np.array(now_pos) - np.array(goal_pos))
        # returning init distance
        return tot_dis / n_objs
    return get_distances(now, goal, cnt_threshold=10, normalize=False)


def read_states(filename):
    import json
    states = json.load(open(filename))
    return states


def process_img(img, args):
    cut = int(args.size * (args.zoom - 1.0) / 2)
    img = np.asarray(img.resize((args.size, args.size)).crop((cut, cut, args.size - cut, args.size - cut)).resize(
        (args.size, args.size)), dtype=np.float32) / 127.5 - 1.0
    img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
    return img


def read_img(img_path, args):
    img = Image.open(img_path)
    img = process_img(img, args)
    return img


def onehot_act(act):
    ret = np.zeros((4), dtype=np.float32)
    ret[act] = 1.
    return ret


def MPC_eval_a_traj(frame0, frame1, frame2, model, act2mat, traj, device, l2=False):
    min_dis = 10000000.0
    now = frame1
    prev_now = frame0
    for act in traj:
        # get prediction
        if args.model_type == 'mood':
            outputs = model.forward(prev_now, now, None, torch.tensor(act2mat[int(act.numpy())]).to(device))
        elif args.model_type == 'edcnn':
            act = torch.from_numpy(onehot_act(act)).to(device)
            outputs = model.forward(prev_now, now, act.unsqueeze(0))
        elif args.model_type == 'clasp':
            action_mat = torch.from_numpy(act2mat[act.item()]).to(device).unsqueeze(0)
            outputs = model.forward(prev_now, now, None, action_mat)
        elif args.model_type == 'wmaefcn':
            # resize the images due to implementation concerns
            prev_now = F.interpolate(prev_now, size=(128, 128), mode='bilinear', align_corners=False)
            now = F.interpolate(now, size=(128, 128), mode='bilinear', align_corners=False)

            act = torch.from_numpy(onehot_act(act)).to(device)
            outputs = model.forward_test(prev_now, now, act.unsqueeze(0))

            prev_now = F.interpolate(prev_now, size=(64, 64), mode='bilinear', align_corners=False)
            now = F.interpolate(now, size=(64, 64), mode='bilinear', align_corners=False)

        # update
        prev_now = now
        if args.model_type == 'mood':
            now = outputs['pred']
        else:
            now = outputs

    # calculate metric
    if l2:
        dis = float(F.mse_loss(frame2, now))
    if args.mixed:
        l2 = float(F.mse_loss(frame2, now))
        pos_err = get_distances(now.squeeze().permute(1, 2, 0), frame2.squeeze().permute(1, 2, 0))
        dis = l2 * 5 + pos_err
    else:
        dis = get_distances(now.squeeze().permute(1, 2, 0), frame2.squeeze().permute(1, 2, 0))

    # log
    if dis < min_dis:
        min_dis = dis

    return min_dis


def MPC_get_a_step(frame0, frame1, frame2, model, act2mat, device):
    num_iter = args.num_iter
    len_path = args.len_path
    num_selected = 10
    sum_tra = args.sum_tra if args.sampled else len(list(product(list(range(4)), repeat=len_path)))
    len_act = len(act2mat)
    distance = np.zeros(sum_tra)
    dist = Categorical(torch.tensor([len_act * [1. / len_act] for i in range(len_path)]))
    for idx_iter in range(num_iter):
        if args.sampled:
            sample_traj = dist.sample((sum_tra,))
        else:
            sample_traj = list(product(list(range(4)), repeat=len_path))
            sample_traj = torch.tensor(sample_traj)
            rand_idx = torch.randperm(sum_tra)
            sample_traj = sample_traj[rand_idx]
        for idx_tra in range(sum_tra):
            distance[idx_tra] = MPC_eval_a_traj(frame0, frame1, frame2, model, act2mat, sample_traj[idx_tra], device, l2=args.l2)

        sorted_args = distance.argsort()
        if idx_iter == num_iter - 1:
            break
        selected_samples = sample_traj[sorted_args][:num_selected]
        new_dist = []
        for idx_path in range(len_path):
            new_dist_sub = [0 for i in range (len_act)]
            for j in range(num_selected):
                new_dist_sub[selected_samples[j, idx_path]] += 1. / num_selected
            new_dist.append(new_dist_sub)
        dist = Categorical(torch.tensor(new_dist))
    # try not to do conjugate actions
    idx_select = sorted_args[0]

    min_dis = distance[idx_select]

    act = sample_traj[idx_select, 0]
    if args.model_type == 'mood':
        gen_image = model.forward(frame0, frame1, None, torch.tensor(act2mat[int(act.numpy())]).to(device))['pred']
    elif args.model_type == 'edcnn':
        act = torch.from_numpy(onehot_act(act.item())).to(device)
        gen_image = model.forward(frame0, frame1, act.unsqueeze(0))
    elif args.model_type == 'clasp':
        action_mat = torch.from_numpy(act2mat[act.item()]).to(device).unsqueeze(0)
        gen_image = model.forward(frame0, frame1, None, action_mat)
    elif args.model_type == 'wmaefcn':
        frame0 = F.interpolate(frame0, size=(128, 128), mode='bilinear', align_corners=False)
        frame1 = F.interpolate(frame1, size=(128, 128), mode='bilinear', align_corners=False)

        act = torch.from_numpy(onehot_act(act.item())).to(device)
        gen_image = model.forward_test(frame0, frame1, act.unsqueeze(0))

        frame0 = F.interpolate(frame0, size=(64, 64), mode='bilinear', align_corners=False)
        frame1 = F.interpolate(frame1, size=(64, 64), mode='bilinear', align_corners=False)

    return sample_traj[idx_select, 0].item(), min_dis, (gen_image + 1) / 2


def tensor2np(img, size):
    output = img.clone()
    output = output.view(3, size, size).cpu().detach().numpy()
    return output


def get_state(idx, states):
    state = {k: v[idx] for k, v in states.items()}
    return state


def main(args):
    if args.gpu == '-1':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))

    # print("translate: ", args.translate_only)
    if args.model_type == 'mood':
        model = Model(map_size=args.map_size, img_size=args.size, num_maps=args.num_maps, n_agent=args.n_agent,\
                    translate_only=args.translate_only).to(device)
    elif args.model_type == 'edcnn':
        from baselines.edcnn.models import Model as edcnn
        model = edcnn(act_space=4, img_size=args.size).to(device)

    # elif args.model_type == 'cswm':
    #     # TODO
    #     from baselines.cswm import modules
    #     model = modules.ContrastiveSWM(
    #         embedding_dim=args.embedding_dim,
    #         hidden_dim=args.hidden_dim,
    #         action_dim=args.action_dim,
    #         input_dims=input_shape,
    #         num_objects=args.num_objects,
    #         sigma=args.sigma,
    #         hinge=args.hinge,
    #         ignore_action=args.ignore_action,
    #         copy_action=args.copy_action,
    #         encoder=args.encoder).to(device)

    elif args.model_type == 'wmaefcn':
        from baselines.wmaefcn.models import WM
        model = WM(True, 4).to(device)

    elif args.model_type == 'clasp':
        from baselines.clasp.models import Model as clasp
        model = clasp(z_dim=16, img_size=args.size).to(device)

    # print("loading from: ", args.test_model_path)

    tmp = torch.load(args.test_model_path, map_location=device)
    model.load_state_dict(tmp)

    model.eval()
    # print('start ...')

    pth = os.path.join('datagen/demonstration', 'roboarm')
    with open(os.path.join(pth, 'demo.txt'), 'r') as f:
        img_pairs, actions = [], []
        for line in f.readlines():
            l = line.split()
            img_pairs.append((l[0], l[1]))
            actions.append(int(l[2]))

    # print(img_pairs, actions)
    if args.model_type == 'mood':
        act2mats = oneshot_act2mats(img_pairs, actions, model, device, args)
        act2mat = act2mats[0]
    elif args.model_type == 'clasp':
        act2mat = clasp_act2mats(img_pairs, actions, model, device, args)
    else:
        act2mat = torch.arange(4).to(device)

    if not os.path.exists(args.test_save_path):
        os.mkdir(args.test_save_path)
    if not os.path.exists(os.path.join(args.test_save_path, 'img')):
        os.mkdir(os.path.join(args.test_save_path, 'img'))
    if not os.path.exists(os.path.join(args.test_save_path, 'states')):
        os.mkdir(os.path.join(args.test_save_path, 'states'))
    if not os.path.exists(os.path.join(args.test_save_path, 'acc')):
        os.mkdir(os.path.join(args.test_save_path, 'acc'))

    env = Roboarm()
    action_space = list(RANDOM_MOVES.values())

    n_objs = 6
    total_steps = args.total_steps
    total_tests = 1
    # print("test set length: ", total_tests)

    acc_observations = np.zeros((total_tests, total_steps + 1))
    acc_states = np.zeros((total_tests, total_steps + 1))
    idx_mpc = args.idx_mpc

    qpos_init = env.get_positions()
    qpos_goal = np.array(json.load(open(f'{args.data_path}/%.6d_goal.json' % idx_mpc)))

    frame0 = read_img(os.path.join(args.data_path, '%.6d_init0.png' % idx_mpc), args)
    frame0 = frame0.view(1, 3, args.size, args.size).to(device)
    frame1 = sensors.capture(idx_mpc, 0).resize((args.size, args.size))
    frame1 = process_img(frame1, args)
    frame1 = frame1.view(1, 3, args.size, args.size).to(device)
    frame2 = read_img(os.path.join(args.data_path, '%.6d_goal.png' % idx_mpc), args)
    frame2 = frame2.view(1, 3, args.size, args.size).to(device)

    Image.fromarray(frame2img(tensor2np(frame0, args.size))).save(
        os.path.join(args.test_save_path, 'img', '{}_pred_0.png'.format(idx_mpc)))
    Image.fromarray(frame2img(tensor2np(frame1, args.size))).save(
        os.path.join(args.test_save_path, 'img', '{}_pred_1.png'.format(idx_mpc)))
    Image.fromarray(frame2img(tensor2np(frame1, args.size))).save(
        os.path.join(args.test_save_path, 'img', '{}_a.png'.format(idx_mpc)))
    Image.fromarray(frame2img(tensor2np(frame2, args.size))).save(
        os.path.join(args.test_save_path, 'img', '{}_gt.png'.format(idx_mpc)))
    Image.fromarray(frame2img(tensor2np(frame2, args.size))).save(
        os.path.join(args.test_save_path, 'img', '{}_pred_{}_gt.png'.format(idx_mpc, total_steps + 1)))

    img1 = frame2img(tensor2np(frame1, args.size))
    img2 = frame2img(tensor2np(frame2, args.size))


    # distance from init to goal
    acc_observations[0, 0] = compute_avg_distance(img1, img2)
    acc_states[0, 0] = compute_avg_distance(qpos_init, qpos_goal, GT=True)

    # print(acc_observations[0])
    # print(acc_states[0])

    with open(os.path.join(args.test_save_path, 'states', 'state_{}.txt'.format(idx_mpc)), 'w') as fs:
        for idx_step in trange(total_steps):
            # print("idx_step: ", idx_step)
            with torch.no_grad():
                ret_act, ret_dis, gen_img = MPC_get_a_step(frame0, frame1, frame2, model, act2mat, device)

            env.step(action_space[ret_act])
            frame0 = frame1
            frame1 = sensors.capture(idx_mpc, idx_step).resize((args.size, args.size))
            frame1.save(os.path.join(args.test_save_path, 'img', '{}_pred_{}.png'.format(idx_mpc, idx_step + 2)))
            save_image(gen_img, os.path.join(args.test_save_path, 'img', '{}_gen_{}_{}.png'.format(idx_mpc, idx_step + 2, ret_act)))
            frame1 = process_img(frame1, args)
            frame1 = frame1.view(1, 3, args.size, args.size).to(device)

            img1 = frame2img(tensor2np(frame1, args.size))
            qpos = env.get_positions()

            acc_observations[0, idx_step + 1] = compute_avg_distance(img1, img2)
            acc_states[0, idx_step + 1] = compute_avg_distance(qpos, qpos_goal, GT=True)

                # for x in qpos: fs.write(str(x) + ' ')
                # fs.write('\n')
                # print(idx_mpc, idx_step + 1, ret_act, dis_step_goal)


        # print('#########################################################')
        # print(f'Total time elasped for episode {idx_mpc}: {end - start}')
        # print('#########################################################')

    print(acc_observations)
    print(acc_states)
    env.pr.stop()
    env.pr.shutdown()
    np.save(os.path.join(args.test_save_path, 'acc', f'acc_pos_err_{args.idx_mpc}.npy'), acc_observations)
    np.save(os.path.join(args.test_save_path, 'acc', f'acc_gt_pos_err{args.idx_mpc}.npy'), acc_states)

# python mpc_multi.py --dataset_name roboarm --data_path datagen/roboarm_MPC --size 64 --test_save_path test_mpc --test_model_path checkpoint_s/model_69.pth --num_iter 4 --len_path 5 --sum_tra 50 --seed 0 --model_type mood --gpu 0 --hide_penalty 20 --sampled --idx_mpc 200
if __name__ == '__main__':
    random_int = str(np.random.randint(0, 10000000000000000))
    while os.path.exists(f'temp{random_int}.png'):
        random_int = str(np.random.randint(0, 10000000000000000))
    # torch.random.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)

    if args.plus:
        from models_plus import Model
    else:
        from models import Model

    frequency = 2500
    duration = 1000

    main(args)
    os.remove(f'temp{random_int}.png')
