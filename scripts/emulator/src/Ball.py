from .Vec2D import Vec2D
from .util import clip
import math
import random
import pygame as pg

class Ball:
    def __init__(self, image, window_size):
        self.image = image
        self.width, self.height = self.image.get_size()
        self.radius = (self.width+self.height)/4

        self.size = Vec2D(self.width, self.height)

        self.window_size = window_size

        self.angle = 0

        self.pos = Vec2D(0, 0)
        self.vel = Vec2D(0, 0)

        self.acceleration = 2000

    def update(self, dt):
        C_drag = 0.01
        A_accel = Vec2D(0, self.acceleration)
        A_drag = -C_drag*self.vel
        A_net = A_accel+A_drag

        self.vel += A_net*dt
        self.pos += self.vel*dt

        max_dx = 200
        k = self.vel.x/max_dx
        k = clip(k, -1, 1)
        omega = k * 180 
        self.angle += omega*dt

        self.check_collision()

    def bounce(self, mouse_pos):
        min_bounce_vel = 900
        # max_bounce_vel = 2200
        max_bounce_vel = 1500
        self.vel.y = clip(
            self.vel.y-min_bounce_vel, 
            -max_bounce_vel, -min_bounce_vel)

        horizontal_bounce = 450
        horizontal_random = 150
        horizontal_limit = 1000
        
        x_diff = -(mouse_pos.x-self.pos.x)/self.radius
        dx_constant = x_diff*horizontal_bounce
        dx_random = random.randint(-horizontal_random, horizontal_random)

        self.vel.x +=  dx_constant + dx_random
        self.vel.x = clip(self.vel.x, -horizontal_limit, +horizontal_limit)

    @property
    def is_out_of_window(self):
        if self.pos.y-self.radius*5 > self.window_size.y:
            return True
        return False

    def check_collision(self):
        if self.pos.x-self.radius < 0:
            self.pos.x = self.radius
            self.vel.x = abs(self.vel.x)
        elif self.pos.x+self.radius > self.window_size.x:
            self.pos.x = self.window_size.x-self.radius
            self.vel.x = -abs(self.vel.x)
    
    def reset(self):
        self.vel.set(0, 0)
        self.angle = 0

    def render(self, surface):
        rot_image = pg.transform.rotate(self.image, self.angle)

        rect = rot_image.get_rect()
        rect.center = self.pos.cast_tuple(int)

        surface.blit(rot_image, rect)