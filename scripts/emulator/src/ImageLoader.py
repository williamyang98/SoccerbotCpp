import os
import pygame as pg

class ImageLoader:
    def __init__(self, directory):
        self.directory = directory

        # self.ball = self.load("ball.png")
        self.ball = self.load("ball_v2.png")
        self.emotes = [self.load(f"emote{i}.png") for i in range(1,6)]
        self.success = [self.load(f"success{i}.png") for i in range(1, 6)]
    
    def load(self, filename):
        return pg.image.load(os.path.join(self.directory, filename)).convert_alpha()