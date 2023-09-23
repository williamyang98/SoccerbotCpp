import numpy as np
import math
import os
import random
from PIL import Image, ImageDraw, ImageFont

def create_light_beam(canvas, position, angle, spread, colour):
    width, height = canvas.size
    beam_length = width+height 
    x, y = position
    x1, y1 = int(x + beam_length*math.cos(angle+spread)), int(y + beam_length*math.sin(angle+spread))
    x2, y2 = int(x + beam_length*math.cos(angle-spread)), int(y + beam_length*math.sin(angle-spread))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.polygon([(x, y), (x1, y1), (x2, y2)], fill=colour)

def create_firework(canvas, pos, explosion_size, colour, streak_size=3, total_lines=12, collapse_chance=0.7):
    draw = ImageDraw.Draw(canvas, "RGBA")
    
    explosion_collapsed = random.random() < collapse_chance

    x, y = pos
    delta = 2*math.pi/total_lines
    angle = 0
    
    if not explosion_collapsed:
        fire_radius = random.randint(0, int(explosion_size/2))
        draw.ellipse([(x-fire_radius, y-fire_radius), (x+fire_radius, y+fire_radius)], fill=colour)
        return

    fire_radius = random.randint(0, int(explosion_size/3))
    start_radius = explosion_size/2-fire_radius
    streak_length = explosion_size-fire_radius
    end_radius = start_radius+streak_length
    if fire_radius > explosion_size/10:
        draw.ellipse([(x-fire_radius, y-fire_radius), (x+fire_radius, y+fire_radius)], fill=colour)

    for _ in range(total_lines):
        x1, y1 = int(x + start_radius*math.cos(angle)), int(y + start_radius*math.sin(angle))
        x2, y2 = int(x + end_radius*math.cos(angle)), int(y + end_radius*math.sin(angle))
        draw.line([(x1,y1), (x2,y2)], fill=colour, width=streak_size)
        angle += delta

def create_ui(canvas, ui):
    canvas.paste(ui, (0, 0), ui)

def create_score(canvas, font, score, fill):
    draw = ImageDraw.Draw(canvas)
    
    score_text = "{0}".format(score)
    _, _, score_width, score_height = draw.textbbox((0,0), score_text, font=font)
    width, height = canvas.size

    x = int(width/2-score_width/2)
    y = int(height/5-score_height/2)
    
    draw.text((x, y), score_text, size=50, font=font, fill=fill)

def populate_emotes(canvas, emotes, total, rect=(0, 0, 1, 1)):
    lower, upper = total
    width, height = canvas.size

    left, top, right, bottom = rect
    left, right = left*width, right*width
    top, bottom = top*height, bottom*height

    total_emotes = random.randint(lower, upper)
    for _ in range(total_emotes):
        emote_img = random.choice(emotes)
        x = int(random.uniform(left, right))
        y = int(random.uniform(top, bottom))
        canvas.paste(emote_img, (x, y), emote_img)
    
# x_centre, y_centre, width, height - normalised to dimensions
def create_ball(canvas, ball, rotation_range):
    background_width, background_height = canvas.size
    ball_width, ball_height = ball.size
    
    # NOTE: Ball can exist outside the visible y-axis, but will always be within the x-axis
    #       PIL pastes the image from the top-left corner
    x = random.randint(0, background_width-ball_width)
    y = random.randint(int(-ball_height/2), int(background_height-ball_height/2))
    rotation = random.randint(rotation_range[0], rotation_range[1])

    canvas.paste(ball.rotate(rotation), (x, y), ball)

    x_centre, y_centre = x+ball_width/2, y+ball_height/2    
    x_centre_norm, y_centre_norm = x_centre/background_width, y_centre/background_height
    width_norm, height_norm = ball_width/background_width, ball_height/background_height

    return (x_centre_norm, y_centre_norm, width_norm, height_norm)