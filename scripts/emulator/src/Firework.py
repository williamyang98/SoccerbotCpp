import pygame as pg
import math
import random
from .Vec2D import Vec2D
from .util import point_rot, clip, get_points

class Firework:
    FLYING = 1
    EXPLODING = 2
    FADING = 3
    FINISHED = 4

    def __init__(self, start, end, ToF, explosion_colour):
        self.start = start
        self.end = end
        self.flight_distance = (start-end).length()
        self.pos = start.copy()

        self.ToF = ToF
        self.elapsed_ToF = 0

        self.vel = (end-start)/ToF
        self.state = Firework.FLYING

        self.explosion_duration = 0.5
        self.elapsed_explosion_duration = 0

        self.fade_duration = 0.5
        self.elapsed_fade_duration = 0

        self.streak_colour = (155, 155, 155)
        self.explosion_colour = explosion_colour
        # self.explosion_colour = (0, 255, 0)

        self.total_rays = random.randint(8, 12)
        self.angles = [i/self.total_rays * math.pi * 2.0 for i in range(self.total_rays)]
        self.ray_dirs = [point_rot(Vec2D(0, 1), alpha) for alpha in self.angles]
    
    def is_finished(self):
        return (self.state == Firework.FINISHED)
    
    def update(self, dt):
        if self.is_finished():
            return
        
        if self.state == Firework.FLYING:
            self.pos += self.vel*dt
            self.elapsed_ToF = clip(self.elapsed_ToF+dt, 0, self.ToF)
            if self.elapsed_ToF >= self.ToF:
                self.state = Firework.EXPLODING
            return
        
        if self.state == Firework.EXPLODING:
            self.elapsed_explosion_duration = clip(
                self.elapsed_explosion_duration+dt, 
                0, self.explosion_duration)
            if self.elapsed_explosion_duration >= self.explosion_duration:
                self.state = Firework.FADING
            return
        
        if self.state == Firework.FADING:
            self.elapsed_fade_duration = clip(
                self.elapsed_fade_duration+dt,
                0, self.fade_duration)
            if self.elapsed_fade_duration >= self.fade_duration:
                self.state = Firework.FINISHED
            return

    def render(self, surface):
        if self.is_finished():
            return

        if self.state == Firework.FLYING:
            self.render_streak(surface)
        elif self.state == Firework.EXPLODING:
            self.render_explosion(surface)
        elif self.state == Firework.FADING:
            self.render_fade(surface)
    
    def render_explosion(self, surface):
        prog = self.elapsed_explosion_duration / self.explosion_duration
        self.render_fireball(surface, prog)
        self.render_rays(surface, prog)
    
    def render_fade(self, surface):
        prog = self.elapsed_fade_duration / self.fade_duration
        alpha = int((1-prog)*255)

        image = pg.Surface(surface.get_size())
        image.set_colorkey((0, 0, 0))
        image.set_alpha(alpha)

        self.render_fireball(image, 1)
        self.render_rays(image, 1)

        surface.blit(image, (0, 0))

    def render_rays(self, surface, prog):
        # center of explosion
        pos = self.end
        max_length = 55

        upper = max_length*prog
        lower = max_length*0.2*prog

        ray_thickness = 5

        for alpha, ray_dir in zip(self.angles, self.ray_dirs):
            dim = Vec2D(ray_thickness, upper-lower)
            ray_pos = pos + ray_dir*(upper-lower)
            points = get_points(ray_pos, alpha, dim)
            points = [p.cast_tuple(int) for p in points]

            pg.draw.polygon(surface, self.explosion_colour, points)
    
    def render_fireball(self, surface, prog):
        # center of explosion
        pos = self.end
        max_explosion_radius = 40
        K_min = 0.25
        K = (1-prog)*(1-K_min) + K_min
        explosion_radius = max_explosion_radius * K

        pg.draw.circle(
            surface, self.explosion_colour,
            pos.cast_tuple(int), int(explosion_radius))
    
    def render_streak(self, surface):
        prog = self.elapsed_ToF / self.ToF
        K = math.cos(prog * math.pi * 2)
        K = K/2 + 0.5
        K = 1-K
        # K = min(0.8, K)
        # K = 1-abs(prog-0.5)*2

        direction = self.end-self.start
        p0 = self.start + direction*prog

        length = K*self.flight_distance*0.25

        p1 = p0 + direction.norm()*-length

        pg.draw.line(
            surface, self.streak_colour, 
            p0.cast_tuple(int), p1.cast_tuple(int), 3)