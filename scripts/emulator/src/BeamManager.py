import pygame as pg
import math
import random

from .Beam import Beam
from .Vec2D import Vec2D

class BeamManager:
    def __init__(self, window_size, max_beams=6):
        self.max_beams = max_beams
        self.window_size = window_size
        self.length = 10*(window_size.x+window_size.y)
        self.beams = []

        self.last_score = 0

        self.min_score = 20
        self.score_delta = 5
        self.last_score = 0

        # self.min_score = 1
        # self.score_delta = 1

        self.left_sub_right = 0
    
    def reset(self):
        self.beams = []
        self.last_score = 0
        self.left_sub_right = 0
    
    def on_score(self, score):
        if score >= self.min_score and\
            score % self.score_delta == 0 and\
            score != self.last_score and\
            len(self.beams) < self.max_beams:
            self.add_beam()

        self.last_score = score
            
    def add_beam(self):
        lower = self.window_size.y * 1.0
        upper = self.window_size.y * 1.2
        y_pos = random.randint(lower, upper)

        x_off = (self.window_size.x * 0.5)

        add_right = self.left_sub_right >= 0

        angle_min = (math.pi/2)*0.4
        angle_max = (math.pi/2)*0.85

        omega = math.pi * (random.random() + 0.3)
        spread = math.pi/70 + ((math.pi/100) * random.random())

        if add_right:
            self.left_sub_right -= 1
            x_pos = self.window_size.x+x_off
            angle_min, angle_max = -angle_max, -angle_min
        else:
            self.left_sub_right += 1
            x_pos = -x_off

        beam = Beam(
            Vec2D(x_pos, y_pos), 
            angle_min, angle_max, omega, spread, 
            self.length)
        
        self.beams.append(beam)

    def update(self, dt):
        for beam in self.beams:
            beam.update(dt)
    
    def render(self, surface):
        for beam in self.beams:
            beam.render(surface)