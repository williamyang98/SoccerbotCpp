from PIL import Image, ImageFont

class GeneratorConfig:
    def __init__(self):
        self.ball_image = None
        self.emote_images = []
        self.background_image = None
        self.score_font = None 
    
    def set_ball_image(self, filepath):
        self.ball_image = Image.open(filepath)
    
    def set_emote_images(self, filepaths):
        images = []
        for filepath in filepaths:
            image = Image.open(filepath)
            images.append(image)
        self.emote_images = images
    
    def set_background_image(self, filepath):
        self.background_image = Image.open(filepath)
    
    def set_score_font(self, font_name, size):
        self.score_font = ImageFont.truetype(font_name, size)
    
    def assert_valid(self):
        assert(self.ball_image != None)
        assert(self.emote_images)
        assert(self.background_image != None)
        assert(self.score_font != None)