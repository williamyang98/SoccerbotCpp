from .Vec2D import Vec2D
from .util import clip
import pygame as pg

class Emote:
    POPPING = 1
    STATIC = 2
    FADING = 3
    EXPIRED = 4

    def __init__(self, image, pos):
        # self.image = image.copy()
        self.image = image
        self.width, self.height = self.image.get_size()
        self.size = Vec2D(self.width, self.height)

        self.original_pos = pos.copy()
        self.pos = pos.copy()

        self.pop_duration = 0.2
        self.pop_distance = 40
        self.static_duration = 0.5
        self.fade_duration = 0.25 
        self.fade_distance = 150

        self.curr_duration = 0
        self.current_state = Emote.POPPING

        self.alpha = 0
    
    def update(self, dt):
        if self.current_state == Emote.EXPIRED:
            return

        self.curr_duration += dt
        if self.current_state == Emote.POPPING:
            prog = clip(self.curr_duration/self.pop_duration, 0, 1)
            self.alpha = int(prog*255)
            self.pos.y = self.original_pos.y - self.pop_distance * prog
            if self.curr_duration > self.pop_duration:
                self.curr_duration = self.pop_duration-self.curr_duration
                self.current_state = Emote.STATIC
        elif self.current_state == Emote.STATIC:
            self.alpha = 255
            self.pos.y = self.original_pos.y - self.pop_distance
            if self.curr_duration > self.static_duration:
                self.curr_duration = self.static_duration-self.curr_duration
                self.current_state = Emote.FADING
        elif self.current_state == Emote.FADING:
            prog = clip(self.curr_duration/self.fade_duration, 0, 1)
            self.alpha = int((1-prog)*255)
            self.pos.y = self.original_pos.y - self.pop_distance - self.fade_distance * prog
            if self.curr_duration > self.fade_duration:
                self.current_state = Emote.EXPIRED
    
    def render(self, surface):
        if self.current_state == Emote.EXPIRED:
            return
        # self.image.set_alpha(self.alpha)
        pos = (self.pos-self.size/2).cast_tuple(int)

        source = self.image.copy()

        image = pg.Surface(source.get_rect().size, pg.SRCALPHA)
        image.fill((255, 255, 255, self.alpha))
        source.blit(image, (0, 0), special_flags=pg.BLEND_RGBA_MULT)
        surface.blit(source, pos)