import pygame as pg
import os
import math
import random

from src import Emulator, Vec2D

ASSETS_DIR = "../assets/"

def main():

    SCREEN_WIDTH = 322
    SCREEN_HEIGHT = 455

    pg.init()
    surface = pg.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

    emulator = Emulator(ASSETS_DIR, Vec2D(SCREEN_WIDTH, SCREEN_HEIGHT))


    running = True
    paused = False
    clock = pg.time.Clock()

    while running:
        frame_ms = clock.tick(60)
        dt = frame_ms*1e-3

        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_p:
                    paused = not paused
            elif ev.type == pg.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if not paused:
                        emulator.on_click(*ev.pos)

        if not paused:
            emulator.update(dt)
        surface.fill((255,255,255))
        emulator.render(surface)
        pg.display.flip()
    
    pg.quit()

    with open("emulator.log", "a") as fp:
        fp.write("[session begin]\n")
        fp.write(f"deaths: {emulator.total_deaths}\n")
        fp.write(f"highscore: {emulator.highscore}\n")
        fp.write(f"scores: {','.join(map(str, emulator.scores))}\n")
        fp.write(f"clicks: {','.join(map(str, emulator.all_clicks))}\n")
        fp.write("\n")


if __name__ == '__main__':
    main()

    

