import pygame as pg
import math
from .Vec2D import Vec2D
from .util import point_rot, clip

class Beam:
    def __init__(self, pos, angle_min, angle_max, omega, spread, length):
        self.pos = pos
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.omega = abs(omega)
        self.spread = spread
        self.length = length

        self.colour = (255, 196, 0)
        self.alpha = 100

        self.left = point_rot(Vec2D(0, -length), -spread/2)
        self.right = point_rot(Vec2D(0, -length), spread/2)

        self.curr_angle = (angle_max+angle_min)/2
        self.direction = 1
    
    def update(self, dt):
        center_err = abs((self.angle_min+self.angle_max)/2 - self.curr_angle)
        norm_center_err = center_err/abs((self.angle_max-self.angle_min)/2)

        min_k = 0.05
        k = (1-norm_center_err)*(1-min_k) + min_k 

        shaped_omega = k*self.direction*self.omega

        self.curr_angle = clip(
            self.curr_angle+shaped_omega*dt,
            self.angle_min,
            self.angle_max)

        if self.curr_angle == self.angle_min:
            self.direction = 1
        elif self.curr_angle == self.angle_max:
            self.direction = -1
    
    def render(self, surface):
        p0 = self.pos
        p1 = self.pos+point_rot(self.left, self.curr_angle)
        p2 = self.pos+point_rot(self.right, self.curr_angle)
        points = [p0, p1, p2]
        points = [p.cast_tuple(int) for p in points]

        image = pg.Surface(surface.get_size())
        image.set_colorkey((0, 0, 0))
        image.set_alpha(self.alpha)
        pg.draw.polygon(image, self.colour, points)

        surface.blit(image, (0, 0))

        

