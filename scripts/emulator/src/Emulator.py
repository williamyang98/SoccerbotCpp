import pygame as pg
import os
import random
import math

from .Vec2D import Vec2D
from .Ball import Ball
from .EmoteManager import EmoteManager
from .HighScoreCounter import HighScoreCounter
from .ScoreCounter import ScoreCounter
from .BeamManager import BeamManager
from .FireworkManager import FireworkManager

from .ImageLoader import ImageLoader

from .util import clip

class Emulator:
    def __init__(self, assets_dir, window_size):
        self.assets_dir = assets_dir
        self.icons_dir = os.path.join(assets_dir, "icons")
        self.fonts_dir = os.path.join(assets_dir, "fonts")
        self.font_filepath = os.path.join(self.fonts_dir, "segoeuil.ttf")

        self.playing = False 
        self.score = 0
        self.total_clicks = 0
        self.highscore = 0
        self.total_deaths = 0
        self.scores = []
        self.all_clicks = []

        self.window_size = window_size

        self.images = ImageLoader(self.icons_dir)
        self.ball = Ball(self.images.ball, self.window_size)

        self.high_score_counter = HighScoreCounter(self.font_filepath, 18)
        self.high_score_counter.pos = Vec2D(window_size.x-12, 50)

        self.score_counter = ScoreCounter(self.font_filepath)
        self.score_counter.pos = Vec2D(window_size.x//2, 65)

        self.emote_manager = EmoteManager(self.images)
        self.beam_manager = BeamManager(window_size)
        self.firework_manager = FireworkManager(window_size, max_fireworks=6)

        self.ball_spawn = Vec2D(window_size.x//2, window_size.y-self.ball.radius-10)
        self.on_fail()
        # to remove erroneous addition
        self.scores = []
        self.all_clicks = []

    def update(self, dt):
        self.emote_manager.update(dt)
        self.beam_manager.update(dt)
        self.firework_manager.update(dt)

        if not self.playing:
            return

        self.ball.update(dt)
        if self.ball.is_out_of_window:
            self.on_fail()

    def render(self, surface):
        self.high_score_counter.render(surface)
        self.score_counter.render(surface)

        self.firework_manager.render(surface)
        self.beam_manager.render(surface)

        self.ball.render(surface)
        self.emote_manager.render(surface)
    
    def on_fail(self):
        self.playing = False
        self.ball.reset()
        self.ball.pos = self.ball_spawn.copy()

        self.total_deaths += 1
        self.scores.append(self.score)
        self.all_clicks.append(self.total_clicks)

        self.highscore = max(self.score, self.highscore)

        self.high_score_counter.score = self.highscore
        self.score_counter.set_state(self.highscore, self.playing)
        self.score = 0
        self.total_clicks = 0

        self.beam_manager.reset()
        self.firework_manager.reset()

    def on_click(self, x, y):
        mouse_pos = Vec2D(x, y)
        ball = self.ball
        dist = (mouse_pos-ball.pos).length()
        missed = dist > ball.radius

        self.spawn_emote(mouse_pos, missed)

        self.total_clicks += 1

        if missed:
            return

        self.playing = True
        self.score += 1

        self.ball.bounce(mouse_pos)
        self.score_counter.set_state(self.score, self.playing)
        self.beam_manager.on_score(self.score)
        self.firework_manager.on_score(self.score)
    
        
    def spawn_emote(self, pos, missed):
        max_random_x = 5
        max_random_y = 5
        emote_random_delta = Vec2D(
            random.randint(-max_random_x, max_random_x), 
            random.randint(-max_random_y, max_random_y))
        emote_pos = pos + emote_random_delta

        if missed:
            self.emote_manager.create_emote(emote_pos)
        else:
            self.emote_manager.create_success(emote_pos)
    
    