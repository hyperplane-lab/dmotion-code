# -*- coding: utf-8 -*-

import numpy as np
from utils import draw_circle, draw_rect, draw_triangle
from tqdm import trange
from PIL import Image
import random
import os

# agent: green objects: red, purple, yellow, blue
# id: 1,2,3,4,5
class Scene(object):
    def __init__(self):
        self.matrix = np.zeros((5,5), dtype=np.int)
        self.type = ['square', 'circle', 'circle', 'triangle', 'triangle']
        self.color = ['rgb(0,255,0)', 'rgb(255,0,0)', 'rgb(255,0,255)', 'rgb(0,0,255)', 'rgb(255,255,0)']
        self.reset()
        
    def reset(self):
        self.matrix = np.zeros((5,5), dtype=np.int)
        p = random.sample(range(0,5*5), 5)
        for i in range(5):
            self.matrix[p[i] // 5][p[i] % 5] = i+1
        self.x = p[0] // 5
        self.y = p[0] % 5
    
    def random_action(self):
        while True:
            action = np.random.randint(0,4)
            if (self.x == 0 and action == 0) or (self.x == 4 and action == 1) or (self.y == 0 and action == 2) \
            or (self.y == 4 and action == 3):
                continue
            else:
                return action
    
    def step(self, action): # action 0,1,2,3
        assert action >= 0 and action <= 3
        if (self.x == 0 and action == 0) or (self.x == 4 and action == 1) or (self.y == 0 and action == 2) \
        or (self.y == 4 and action == 3):
            raise NotImplementedError
        cur_obj = 0
        cur_pos = [self.x, self.y]
        if action == 0:
            direction = [-1,0]
        elif action == 1:
            direction = [1,0]
        elif action == 2:
            direction = [0,-1]
        elif action == 3:
            direction = [0,1]
        while cur_pos[0] >= 0 and cur_pos[0] < 5 and cur_pos[1] >= 0 and cur_pos[1] < 5:
            tmp = self.matrix[cur_pos[0],cur_pos[1]]
            self.matrix[cur_pos[0],cur_pos[1]] = cur_obj
            cur_obj = tmp
            cur_pos[0] += direction[0]
            cur_pos[1] += direction[1]
            if cur_obj == 0:
                break
        self.x += direction[0]
        self.y += direction[1]
    
    def render(self):
        img = Image.new('RGB', (500, 500), 'black')
        for i in range(5):
            for j in range(5):
                if self.matrix[i,j] == 0:
                    continue
                else:
                    shape = self.type[self.matrix[i,j]-1]
                    color = self.color[self.matrix[i,j]-1]
                    if shape == 'square':
                        draw_rect(img, (100,100), (j*100, i*100), color)
                    elif shape == 'circle':
                        draw_circle(img, 50, (j*100+50, i*100+50), color)
                    elif shape == 'triangle':
                        draw_triangle(img, [(j*100+50, i*100), (j*100, i*100+100), (j*100+100, i*100+100)], color)
                    else:
                        raise NotImplementedError
        return img.resize((128,128))
                
env = Scene()
n_frames = 32
if not os.path.exists('datagen/gridworld'):
    os.mkdir('datagen/gridworld')
    
with open('datagen/gridworld/data.txt', 'w') as f:
    for i_episode in trange(700):
        env.reset()
        for t in range(n_frames):
            img = env.render()
            img.save('datagen/gridworld/%.6d_%.2d.png' % (i_episode, t))
            action = env.random_action()
            f.write('%.6d_%.2d %d\n' % (i_episode, t, action))
            env.step(action)
