from collections import deque
import random
from .Emote import Emote

class EmoteManager:
    def __init__(self, images):
        self.images = images
        self.emotes = deque([])
    
    def create_success(self, pos):
        image = random.choice(self.images.success)
        self.emotes.append(Emote(image, pos))
    
    def create_emote(self, pos):
        image = random.choice(self.images.emotes)
        self.emotes.append(Emote(image, pos))
    
    def update(self, dt):
        for emote in self.emotes:
            emote.update(dt)
        
        while len(self.emotes) > 0 and self.emotes[0].current_state == Emote.EXPIRED:
            self.emotes.popleft()
    
    def render(self, surface):
        for emote in self.emotes:
            emote.render(surface)