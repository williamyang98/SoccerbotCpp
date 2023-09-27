from PIL import Image, ImageFont

class GeneratorConfig:
    def __init__(self):
        self.ball_image = None
        self.emote_images = []
        self.background_image = None
        self.score_font = None 
    
    def set_ball_image(self, filepath):
        self.ball_image = self._load_image(filepath)
    
    def set_emote_images(self, filepaths):
        images = []
        for filepath in filepaths:
            image = self._load_image(filepath)
            images.append(image)
        self.emote_images = images
    
    def set_background_image(self, filepath):
        self.background_image = self._load_image(filepath)
    
    def set_score_font(self, font_name, size):
        self.score_font = ImageFont.truetype(font_name, size)

    def _load_image(self, filepath):
        image = Image.open(filepath)
        image.load()
        return image
    
    def assert_valid(self):
        assert(self.ball_image != None)
        assert(self.emote_images)
        assert(self.background_image != None)
        assert(self.score_font != None)