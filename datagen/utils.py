# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw
import numpy as np
import os
from pathlib import Path
from glob import glob

def draw_square(img, side, center_pos, color = 'red'):
    draw = ImageDraw.Draw(img)
    # x0, y0, x1, y1
    draw.rectangle((center_pos[0]-side//2, center_pos[1]-side//2, center_pos[0]+side//2, center_pos[1]+side//2), color)
    # img.show()

def draw_rect(img, size, pos, color='red'):
    draw = ImageDraw.Draw(img)
    draw.rectangle((pos[0], pos[1], pos[0] + size[0], pos[1] + size[1]), color)

def draw_circle(img, radius, center_pos, color = 'red'):
    draw = ImageDraw.Draw(img)
    # x0, y0, x1, y1
    draw.ellipse((center_pos[0]-radius, center_pos[1]-radius, center_pos[0]+radius, center_pos[1]+radius), color)
    # img.show()

def draw_triangle(img, pts, color='red'):
    draw = ImageDraw.Draw(img)
    # x0, y0, x1, y1
    draw.polygon(pts, fill = color)
    # img.show()

def paste_img(img, target, xidx, yidx, size=128):
    target.paste(img, (xidx * size, yidx * size))
    

def draw_flow(flow, sz, pos, motion, geo):
    pixels = Image.new('L', (flow.shape[0], flow.shape[1]), 'black')
    draw = ImageDraw.Draw(pixels)
    if geo == 'rect':
        draw.rectangle((pos[0], pos[1], pos[0] + sz[0], pos[1] + sz[1]), 'white')
    elif geo == 'circle':
        draw.ellipse((pos[0]-sz, pos[1]-sz, pos[0]+sz, pos[1]+sz), 'white')
    
    pixels = np.asarray(pixels)
    mask = (pixels > 0)
    
    flow *= (np.stack([mask] * 2, -1) == 0)
    for axis in range(2):
        flow[..., axis] += mask * motion[axis]
    
    return flow
    
def random_uniform_in_ranges(ranges):
    '''ranges = [(a1,b1),(a2,b2),...,(an,bn)]'''
    low, up = 9999999, -9999999
    for i in ranges:
        assert i[0] <= i[1]
        low = min(low, i[0])
        up = max(up, i[1])
        
    while True:
        ret = np.random.uniform(low, up)
        flag = False
        for i in ranges:
            if ret >= i[0] and ret <= i[1]:
                flag = True
                break
        if flag:
            return ret
     
def random_int_uniform_in_ranges(ranges):
    '''ranges = [(a1,b1),(a2,b2),...,(an,bn)]'''
    low, up = 9999999, -9999999
    for i in ranges:
        assert i[0] <= i[1]
        low = min(low, i[0])
        up = max(up, i[1])
        
    while True:
        ret = np.random.randint(low, up)
        flag = False
        for i in ranges:
            if ret >= i[0] and ret <= i[1]:
                flag = True
                break
        if flag:
            return ret

def make_or_clean_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    for f in glob(f'{path}/*'):
        os.remove(f)
