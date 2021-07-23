"""
https://github.com/stepjam/PyRep/blob/master/examples/example_panda_reach_target.py
"""
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
from pyrep.objects.vision_sensor import VisionSensor
from tqdm import tqdm
from os.path import dirname, join, abspath
import numpy as np
from random import shuffle
from PIL import Image
import math
from datagen.utils import make_or_clean_dir

import argparse


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


def items_on_table(objects):
    for obj in objects.values():
        if obj.shape.get_position()[-1] < 0.7:
            return False
    return True


class VisionSensors:
    """A list of vision sensors in PyRep for capturing snapshots"""
    def __init__(self, sensor_names):
        self.sensors = {name: VisionSensor(name) for name in sensor_names}

    def capture(self, episode, timestep):
        for name, cam in self.sensors.items():
            view = name.split('_')[-1]
            img = Image.fromarray((cam.capture_rgb() * 255).astype(np.uint8))
            if 'side' in name:
                img = img.rotate(180, expand=True).resize((IMG_SIZE))
            else:
                img = img.rotate(270, expand=True).resize((IMG_SIZE))
            # img.save(f'{DATA_PATH}/{view}_{episode}_{timestep}.png')
            # img.save(f'{DATA_PATH}/%.6d_%.2d.png' % (episode, timestep))
            return img


def chase_objects(objects, agent):
    if i == 0:
        choice = 'target'
    else:
        p = max(0.99 ** i, 0.3)
        choice = np.random.choice(['random', 'target'], p=[p, 1 - p])

    # get a random sudo target
    target = np.random.choice(list(objects.values()))

    # try to move arm to target
    target_position = target.shape.get_position()
    if choice == 'random':
        # random perturbatioins
        target_position = [
            target_position[0] + np.random.uniform(-0.1, 0.1),
            target_position[1] + np.random.uniform(-0.1, 0.1),
            target_position[2]]
    return agent.get_linear_path(position=target_position, euler=[0, math.radians(180), 0], steps=2)


def random_moves(last_move):
    p = [0.01, 0.01, 0.01, 0.01]
    if last_move is None:
        next_move = np.random.choice(list(RANDOM_MOVES.keys()))
    elif np.allclose(last_move, RANDOM_MOVES['up']):
        p[0] = 0.97
        next_move = np.random.choice(list(RANDOM_MOVES.keys()), p=p)
    elif np.allclose(last_move, RANDOM_MOVES['down']):
        p[1] = 0.97
        next_move = np.random.choice(list(RANDOM_MOVES.keys()), p=p)
    elif np.allclose(last_move, RANDOM_MOVES['right']):
        p[2] = 0.97
        next_move = np.random.choice(list(RANDOM_MOVES.keys()), p=p)
    elif np.allclose(last_move, RANDOM_MOVES['left']):
        p[3] = 0.97
        next_move = np.random.choice(list(RANDOM_MOVES.keys()), p=p)
    else:
        raise Exception
    return RANDOM_MOVES[next_move], next_move


def get_random_path(gripper, next_move):
    target_position = gripper.get_position()
    if next_move[0]:
        target_position[0] += next_move[0]
    elif next_move[1]:
        target_position[1] += next_move[1]
    target_position[-1] = TABLE_LEVEL
    return agent.get_linear_path(position=target_position, euler=[0, math.radians(180), 0], steps=2)


def match_action(direction):
    return list(RANDOM_MOVES.keys()).index(direction)


def detect_proximity(gripper, objects):
    gripper_pos = gripper.get_position()
    closest_obj, closest_dist = None, np.inf
    for obj in objects.values():
        diff = np.array(gripper_pos) - np.array(obj.position)
        distance = np.linalg.norm(diff, ord=2)
        if distance < closest_dist:
            closest_dist = distance
            closest_obj = obj
        if distance < 0.2:
            return True, closest_obj
    return False, closest_obj


def random_biased_moves(gripper, closest_obj):
    obj_pos = closest_obj.position
    gripper_pos = gripper.get_position()
    diff = np.array(obj_pos) - np.array(gripper_pos)
    if diff[0] > 0 and diff[1] > 0:
        next_move = np.random.choice(['down', 'right'])
    elif diff[0] < 0 and diff[1] > 0:
        next_move = np.random.choice(['up', 'right'])
    elif diff[0] < 0 and diff[1] < 0:
        next_move = np.random.choice(['up', 'left'])
    elif diff[0] > 0 and diff[1] < 0:
        next_move = np.random.choice(['down', 'left'])
    return RANDOM_MOVES[next_move], next_move


def capture_demo(gripper):
    cam = sensors.sensors[SENSORS[0]]
    timestep, episode = 0, 0
    make_or_clean_dir('datagen/demonstration/roboarm')
    with open('datagen/demonstration/roboarm/demo.txt', 'w') as f:
        for i, direction in enumerate(RANDOM_MOVES.values()):
            try:
                path = get_random_path(gripper, direction * 4)
            except:
                return False
            step_size = np.array([0., 0.])
            img = Image.fromarray((cam.capture_rgb() * 255).astype(np.uint8))
            img = img.rotate(270, expand=True).resize((IMG_SIZE))
            img.save(f'datagen/demonstration/roboarm/%.6d_%.2d.png' % (episode, timestep))
            timestep += 1
            position_ = gripper.get_position()[:-1]
            while (abs(step_size[0]) < 0.08 and abs(step_size[1]) < 0.08):
                done = path.step()
                pr.step()
                position = gripper.get_position()[:-1]
                step_size += position - position_
                position_ = position
                if done:
                    path = get_random_path(gripper, direction * 4)
            img = Image.fromarray((cam.capture_rgb() * 255).astype(np.uint8))
            img = img.rotate(270, expand=True).resize((IMG_SIZE))
            img.save(f'datagen/demonstration/roboarm/%.6d_%.2d.png' % (episode, timestep))
            timestep += 1
            f.write(f'%.6d_%.2d.png %.6d_%.2d.png {i}\n' % (episode, timestep - 2, episode, timestep - 1))
    print('Demo captured')
    return True


def close_gripper(gripper):
    done = False
    while not done:
        done = gripper.actuate(0., velocity=1.)
        pr.step()


parser = argparse.ArgumentParser()
parser.add_argument('--eps', type=int, default=20)
parser.add_argument('--headless', action='store_true', default=False)
parser.add_argument('--increment', default=20, type=int)
parser.add_argument('--scene_path', type=str, default='MOOD_scene2.ttt')
parser.add_argument('--img_size', default=64, type=int)
parser.add_argument('--save_scenes', action='store_true', default=False)
args = parser.parse_args()

if args.eps < args.increment:
    raise ValueError ('Number of episodes must be bigger than the number of increments')
if args.eps % args.increment != 0:
    raise ValueError ('Number of episodes must integer divide the number of increments')


SENSORS = ['Vision_sensor']#, 'Vision_sensor_front', 'Vision_sensor_side']
IMG_SIZE = (args.img_size, args.img_size)
EPISODES = args.eps
DATA_PATH = f'datagen/roboarm_{EPISODES // args.increment}'
NUM_MOVES = 400
TABLE_LEVEL = 0.77
RANDOM_MOVES = {
    'up': np.array([-0.01, 0, 0]),
    'down': np.array([0.01, 0, 0]),
    'right': np.array([0, 0.01, 0]),
    'left': np.array([0, -0.01, 0])}
SCENE_FILE = join(dirname(abspath(__file__)), args.scene_path)
make_or_clean_dir(DATA_PATH)
demo_captured = True

pr = PyRep()
pr.launch(SCENE_FILE, headless=args.headless)
pr.start()
agent = Panda()
gripper = PandaGripper()
sensors = VisionSensors(SENSORS)
policy = 'random'

position_min, position_max = [0.3, -0.5, TABLE_LEVEL], [0.8, 0.6, TABLE_LEVEL]
object_area = list(np.random.uniform(position_min, position_max))
random_ranges = [(0.01, 0.06), (-0.05, 0), (0.07, 0.12), (-0.11, -0.04), (0.13, 0.18), (-0.16, -0.1)]

red_cube = RepObject(PrimitiveShape.CUBOID, [0.18, 0.15, 0.11], [1.0, 0.1, 0.1], object_area, random_ranges[0], 'red_cube')
pink_cylinder = RepObject(PrimitiveShape.CYLINDER, [0.13, 0.14, 0.08], [1.0, 0.1, 1.0], object_area, random_ranges[1], 'pink_cylinder')
green_cube = RepObject(PrimitiveShape.CUBOID, [0.17, 0.15, 0.14], [0.1, 1.0, 0.1], object_area, random_ranges[2], 'green_cube')
blue_cylinder = RepObject(PrimitiveShape.CYLINDER, [0.13, 0.15, 0.08], [0.1, 0.1, 1.0], object_area, random_ranges[3], 'blue_cylinder')
yellow_cube = RepObject(PrimitiveShape.CUBOID, [0.14, 0.14, 0.13], [1.0, 1.0, 0.1], object_area, random_ranges[4], 'yellow_cube')
cyan_cylinder = RepObject(PrimitiveShape.CYLINDER, [0.17, 0.15, 0.09], [0.1, 1.0, 1.0], object_area, random_ranges[5], 'cyan_cylinder')

objects = {
    'red_cube': red_cube,
    'pink_cylinder': pink_cylinder,
    'green_cube': green_cube,
    'blue_cylinder': blue_cylinder,
    'yellow_cube': yellow_cube,
    'cyan_cylinder': cyan_cylinder}
starting_joint_positions = agent.get_joint_positions()
close_gripper(gripper)


if __name__ == '__main__':
    episode = EPISODES - args.increment
    states = {}
    with open(f'{DATA_PATH}/data.txt', 'w') as f:
        pbar = tqdm(total=EPISODES-episode)
        while episode < EPISODES:
            ##################
            # INITIALIZATION #
            ##################

            # Reset the arm at the start of each 'episode'
            agent.set_joint_positions(starting_joint_positions)

            # new position for objects every run
            object_area = list(np.random.uniform(position_min, position_max))

            # For each object, place randomly
            shuffle(random_ranges)
            object_violation = False

            for i, obj in enumerate(objects.values()):
                obj.randomize_position(list(np.random.uniform(position_min, position_max)), random_ranges[i])

            # move gripper to random location near items
            try:
                position = [object_area[0] + np.random.choice([np.random.uniform(0.3, 0.4), -np.random.uniform(0.3, 0.4)]),
                            object_area[1] + np.random.choice([np.random.uniform(0.3, 0.4), -np.random.uniform(0.3, 0.4)]),
                            object_area[2]]
                path = agent.get_path(
                    position=position, euler=[0, math.radians(180), 0])
            except ConfigurationPathError as e:
                continue
            done = False
            while not done:
                done = path.step()
                pr.step()

            for obj in objects.values():
                cur_orient = obj.shape.get_orientation()
                obj.shape.set_orientation([np.random.uniform(1e-6, 9e-6), np.random.uniform(-3e-5, -2.5e-5), cur_orient[2]])
                if obj.shape.get_position()[2] > 0.85 or abs(obj.shape.get_orientation()[0]) > 0.1:
                    object_violation = True
            if object_violation or not items_on_table(objects):
                continue

            #############
            # SELF-PLAY #
            #############
            timestep, n_frame = 0, 0
            next_move = None
            flag = False
            step_size = np.array([0., 0.])
            images, actions = [], []
            states[episode] = {}
            states[episode]['gripper'] = [list(agent.get_joint_positions())]
            for obj in objects.values():
                states[episode][f'{obj.name}_p'] = [list(obj.shape.get_position())]
                states[episode][f'{obj.name}_o'] = [list(obj.shape.get_orientation())]
            for i in range(NUM_MOVES):
                try:
                    if policy == 'chase':
                        path = chase_objects(objects, agent)
                        # TODO
                        action = 0
                    elif policy == 'random':
                        proximity, closest_obj = detect_proximity(gripper, objects)
                        if flag:
                            pass
                        elif proximity or i == 0:
                            next_move, direction = random_moves(next_move)
                            if not demo_captured and proximity:
                                demo_captured = capture_demo(gripper)
                        else:
                            next_move, direction = random_biased_moves(gripper, closest_obj)
                        action = match_action(direction)
                        path = get_random_path(gripper, next_move)
                    else:
                        raise Exception
                except ConfigurationPathError as e:
                    continue

                # execute path
                done = False
                # robot arm does not leave table, no z
                position_ = gripper.get_position()[:-1]
                flag = True
                time_steps = 0
                
                while not done and time_steps < 10000:
                    # check if all items are on table
                    if not items_on_table(objects):
                        break
                    done = path.step()
                    pr.step()
                    position = gripper.get_position()[:-1]
                    moved = position - position_
                    # if robot arm moved, then capture data
                    step_size += moved
                    if abs(step_size[0]) > 0.08 or abs(step_size[1]) > 0.08:
                        flag = False
                        timestep += 1
                        image = sensors.capture(episode, timestep)
                        a = '%.6d_%.2d %d\n' % (episode, timestep, int(action))
                        step_size = np.array([0., 0.])
                        images.append(image)
                        actions.append(a)
                        states[episode]['gripper'].append(list(agent.get_joint_positions()))
                        for obj in objects.values():
                            states[episode][f'{obj.name}_p'].append(list(obj.shape.get_position()))
                            states[episode][f'{obj.name}_o'].append(list(obj.shape.get_orientation()))
                    if args.save_scenes:
                        pr.export_scene(f'{DATA_PATH}/%.6d_%.2d\n.ttt' % (episode, timestep))
                    position_ = position
                    n_frame += 1
                    time_steps += 1

            if timestep > 30:
                episode += 1
                pbar.update(1)
                for img, action in zip(images, actions):
                    f.write(action)
                    episode_time = action.split()[0]
                    img.save(f'{DATA_PATH}/{episode_time}.png')
    import json
    json.dump(states, open(f'{DATA_PATH}/states.json', 'w'))
    pr.stop()
    pr.shutdown()
