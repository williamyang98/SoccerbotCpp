from .Vec2D import Vec2D
import pygame as pg

class HighScoreCounter:
    def __init__(self, font_path, size): 
        self.font = pg.font.Font(font_path, size)
        # self.font.set_bold(True)
        self.colour = (0,0,0)
        self.top_text = self.font.render("High Score", True, self.colour)

        self._score = 0 
        self.score_text = self.font.render(f"{self.score}", True, self.colour)

        self.y_diff = 25 

        self.pos = Vec2D(0, 0)

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, score):
        self._score = score
        self.score_text = self.font.render(f"{self._score}", True, self.colour)
    
    def render(self, surface):
        x, y = self.pos.x, self.pos.y

        width, height = self.top_text.get_size()
        top_rect = self.top_text.get_rect()
        top_rect.center = (x-width//2, y)

        width, height = self.score_text.get_size()
        score_rect = self.score_text.get_rect()
        score_rect.center = (x-width//2, y+self.y_diff)

        surface.blit(self.top_text, top_rect)
        surface.blit(self.score_text, score_rect)