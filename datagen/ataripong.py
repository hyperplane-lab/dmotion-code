# -*- coding: utf-8 -*-

import gym
import random
from PIL import Image
from tqdm import trange
import os

class pong(object):
    def __init__(self):
        self.env = gym.make('PongNoFrameskip-v4')
    def reset(self):
        return self.env.reset()
    def close(self):
        self.env.close()
    def render(self):
        self.env.render()
    def step(self, action):
        assert action >= 1 and action <= 3
        self.env.step(action)
        for i in range(7):
            observation, _, _, _ = self.env.step(1)
        return observation
        
env = pong()

if not os.path.exists('datagen/atari-pong'):
    os.mkdir('datagen/atari-pong')

with open('datagen/atari-pong/data.txt', 'w') as f:
    for i_episode in trange(800):
        observation = env.reset()
        for t in range(34):
            action = random.randint(1,3) # action 1,2,3
            if t >= 8:
                im = observation[35:195]
                Image.fromarray(im).resize((128,128)).save('datagen/atari-pong/%.6d_%.2d.png' % (i_episode, t))
                f.write('%.6d_%.2d %d\n' % (i_episode, t, action))
            observation = env.step(action)

env.close()