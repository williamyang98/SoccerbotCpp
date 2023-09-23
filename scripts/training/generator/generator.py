import numpy as np
import math
import random

from .overlays import *

GREY = (100, 100, 100)
BLUE = (0, 135, 255)
YELLOW = (255, 241, 192, 10)

FIREWORK_COLOURS = [
    (86, 213, 77),      # green
    (255, 149, 234),    # pink
    (252, 214, 81),     # yellow
    (79, 228, 255),     # blue
]

class BasicSampleGenerator:
    def __init__(self, config):
        self.config = config
        self.params = {}
        config.assert_valid()

    def create_sample(self):
        size = self.config.background_image.size
        sample = self.create_background(size)
        score = random.randint(0, 60)
        
        if score > 10:
            self.create_light_beams(sample)

        if score > 30:
            self.create_fireworks(sample)

        create_ui(sample, self.config.background_image)

        # create score
        if random.uniform(0, 1) > 0.5:
            text_colour = GREY
        else:
            text_colour = BLUE

        if score > 30:
            score = random.randint(30, 100000)
            text_colour = GREY

        create_score(sample, self.config.score_font, score, text_colour)        

        has_ball = random.randint(0, 1)
        if has_ball:
            bounding_box = create_ball(sample, self.config.ball_image, (0, 360))
        else:
            bounding_box = (0, 0, 0, 0)

        for _ in range(random.randint(0, 4)):
            sample_type = random.uniform(0, 1)
            if sample_type < 0.3:
                rect = self.get_streaked_emotes(bounding_box[:2])
                populate_emotes(sample, self.config.emote_images, total=(0, 25), rect=rect)
            elif sample_type < 0.5:
                rect = self.get_local_scattered_emotes()
                populate_emotes(sample, self.config.emote_images, total=(0, 15), rect=rect)
            elif sample_type < 0.8:
                populate_emotes(sample, self.config.emote_images, total=(0, 10))
            else:
                pass
            
        return (sample, bounding_box, has_ball)

    def create_light_beams(self, sample, x_offset=100, max_light_beams=5):
        width, height = sample.size

        total_light_beams = random.randint(1, max_light_beams)

        spread = math.pi/100
        y = height - 10 

        for _ in range(total_light_beams):
            left = random.random() > 0.5
            x = -x_offset if left else width+x_offset
            angle = random.uniform(-math.pi/2, 0) if left else random.uniform(-math.pi, -math.pi/2)
            create_light_beam(sample, (x, y), angle, spread, YELLOW)


    def create_fireworks(self, sample, max_fireworks=5, colours=FIREWORK_COLOURS):
        width, height = sample.size

        total_fireworks = random.randint(1, max_fireworks)

        for _ in range(total_fireworks):
            explosion_size = random.randint(10, 60)
            # NOTE: Even though we have the main colours for the fireworks in the game they can vary between versions
            #       Also the game may apply a tonemap which may alter the colour slightly, so we add in variation for robustness
            colour = random.choice(colours)
            colour = [c+random.randint(-30, 30) for c in colour]
            colour = [np.clip(c, 0, 255) for c in colour]
            colour = tuple([int(c) for c in colour])

            x, y = random.randint(0, width), random.randint(0, height)
            create_firework(sample, (x, y), explosion_size, colour)


    def get_streaked_emotes(self, pos=None, x_offset=0.1, y_offset=0.1, width=0.05, height=0.3):
        if pos is None:
            x, y = random.uniform(0, 1), random.uniform(0, 1)
        else:
            x, y = pos

        left = x + random.uniform(-x_offset, x_offset)
        top = y + random.uniform(-y_offset, y_offset) - random.uniform(0, height)

        left = np.clip(left, 0, 1)
        top = np.clip(top, 0, 1)
        right = np.clip(left+random.uniform(0 ,width), 0, 1)
        bottom = np.clip(top+random.uniform(0, height), 0, 1)
        return (left, top, right, bottom)


    def get_local_scattered_emotes(self):
        left, top = random.random(), random.random()
        width, height = random.random(), random.random()
        right = np.clip(left+width, 0, 1)
        bottom = np.clip(top+height, 0, 1)
        return (left, top, right, bottom)

    def create_background(self, size, colour=(255, 255, 255, 255)):
        image = Image.new("RGBA", size, colour)
        return image






