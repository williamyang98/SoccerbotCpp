if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Quantize h5f model to tensorflow lite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-in", type=str, default="./models/model_train.h5f", help="Input path for trained model")
    parser.add_argument("--model-out", type=str, default="./models/quant_out.tflite", help="Output path for quantized model")
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    parser.add_argument("--downscale", type=float, default=4, help="Amount to downscale the input by")
    args = parser.parse_args()

    # get the generator config
    from generator import GeneratorConfig, BasicSampleGenerator
    import numpy as np
    import os
    import pathlib
    import glob
    
    pathlib.Path(os.path.dirname(args.model_out)).mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig()
    config.set_background_image(os.path.join(args.asset_path, "icons/blank.png"))
    config.set_ball_image(os.path.join(args.asset_path, "icons/ball_v2.png"))

    emote_filepaths = []
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/success*.png")))
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/emote*.png")))
    config.set_emote_images(emote_filepaths)
    config.set_score_font(os.path.join(args.asset_path, "fonts/segoeuil.ttf"), 92)
    generator = BasicSampleGenerator(config)

    image, bounding_box, has_ball = generator.create_sample()
    image = image.convert("RGB")
    image = image.resize((int(x/args.downscale) for x in image.size))
    im_width, im_height = image.size
    im_channels = 3

    # create the model
    import tensorflow as tf
    def create_model():
        from model import SoccerBotModel
        module = SoccerBotModel()
        x_in = tf.keras.layers.Input(shape=(im_height, im_width, im_channels))
        y_out = module(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        return model
    
    model = create_model()
    model.summary()

    model.load_weights(args.model_in)
    print(f"Loaded weights from '{args.model_in}'")

    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_model = quant_converter.convert()

    with open(args.model_out, "wb+") as fp:
        fp.write(quant_model)
        print(f"saved weights to '{args.model_out}'")
    
    exit(0)