from .Vec2D import Vec2D
import pygame as pg

class ScoreCounter:
    def __init__(self, font_path):
        self.small_font = pg.font.Font(font_path, 18)
        self.large_font = pg.font.Font(font_path, 75)

        self.primary_colour = (0,121,241)
        self.secondary_colour = (128,128,128)

        self.pos = Vec2D(0, 0)
        self.start_text = self.small_font.render("Current Best", True, self.secondary_colour)

        self.set_state(0, False)

        self.y_diff = 65

    def set_state(self, score, started):
        colour = self.primary_colour if not started else self.secondary_colour
        self.score_text = self.large_font.render(f"{score}", True, colour)
        self.started = started
    
    def render(self, surface):
        x, y = self.pos.x, self.pos.y
        if not self.started:
            width, height = self.start_text.get_size()
            rect = self.start_text.get_rect()
            rect.center = (x, y+height//2)
            surface.blit(self.start_text, rect)
            y_off = self.y_diff
        else:
            _, height = self.start_text.get_size()
            y_off =  height

        width, height = self.score_text.get_size()
        rect = self.score_text.get_rect()
        rect.center = (x, y+height//2+y_off)
        surface.blit(self.score_text, rect)