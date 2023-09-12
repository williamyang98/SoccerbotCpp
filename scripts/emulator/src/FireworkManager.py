import pygame as pg
import random
from collections import deque
from .Firework import Firework
from .Vec2D import Vec2D

class FireworkManager:
    def __init__(self, window_size, max_fireworks=6):
        self.window_size = window_size
        self.fireworks = set()
        self.last_score = 0
        self.max_fireworks = max_fireworks
        self.spawn_chance = 0.97

        self.colours = [
            (255, 196, 0),
            (0, 214, 255),
            (252, 100, 255),
            (19, 211, 31)
        ]
    
    def reset(self):
        self.last_score = 0
    
    def on_score(self, score):
        self.last_score = score

    def update(self, dt):
        if self.last_score >= 30 and\
            len(self.fireworks) < self.max_fireworks and\
            random.random() > self.spawn_chance:
        # if len(self.fireworks) < self.max_fireworks and\
        #     random.random() > 0.97:
            self.add_firework()

        alive_fireworks = set()
        for firework in self.fireworks:
            firework.update(dt)
            if not firework.is_finished():
                alive_fireworks.add(firework)

        self.fireworks = alive_fireworks
    
    def add_firework(self):
        y_start = self.window_size.y + 100

        x_start = random.randint(
            int(self.window_size.x*0.1),
            int(self.window_size.x*0.9))

        
        x_end = random.randint(
            int(self.window_size.x*0.05),
            int(self.window_size.x*0.95))

        y_end = random.randint(
            int(self.window_size.y*0.05),
            int(self.window_size.y*0.65))
        
        start = Vec2D(x_start, y_start)
        end = Vec2D(x_end, y_end)
        ToF = (random.random() + 0.5) / 1.5

        explosion_colour = random.choice(self.colours)

        firework = Firework(start, end, ToF, explosion_colour)
        self.fireworks.add(firework)
    
    def render(self, surface):
        for firework in self.fireworks:
            firework.render(surface)