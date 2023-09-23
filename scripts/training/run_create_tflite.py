if __name__ == '__main__':
    import argparse
    from models.select_model import get_model_types, select_model
    MODEL_TYPES = get_model_types()
    DEFAULT_MODEL_TYPE = MODEL_TYPES[0]
    DEFAULT_MODEL_PATH = "./data/model-train-*.h5f"
    DEFAULT_QUANT_PATH = "./data/quant-out-*.tflite"

    parser = argparse.ArgumentParser(description="Quantize h5f model to tensorflow lite", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES, help="Type of model")
    parser.add_argument("--model-in", type=str, default=DEFAULT_MODEL_PATH, help="Input path for trained model. * is replaced with --model-type.")
    parser.add_argument("--model-out", type=str, default=DEFAULT_QUANT_PATH, help="Output path for quantized model. * is replaced with --model-type.")
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    args = parser.parse_args()

    # get the generator config
    from generator import GeneratorConfig, BasicSampleGenerator
    import numpy as np
    import os
    import pathlib
    import glob

    PATH_MODEL_IN = args.model_in.replace("*", args.model_type)
    PATH_MODEL_OUT = args.model_out.replace("*", args.model_type)
    
    pathlib.Path(os.path.dirname(PATH_MODEL_OUT)).mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig()
    config.set_background_image(os.path.join(args.asset_path, "icons/blank.png"))
    config.set_ball_image(os.path.join(args.asset_path, "icons/ball.png"))

    emote_filepaths = []
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/success*.png")))
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/emote*.png")))
    config.set_emote_images(emote_filepaths)
    config.set_score_font(os.path.join(args.asset_path, "fonts/segoeuil.ttf"), 92)
    generator = BasicSampleGenerator(config)

    image, bounding_box, has_ball = generator.create_sample()
    im_original_width, im_original_height = image.size
    im_channels = 3

    # create the model
    import tensorflow as tf
    SoccerBotModel = select_model(args.model_type)
    DOWNSCALE_RATIO = SoccerBotModel.DOWNSCALE_RATIO
    im_downscale_width, im_downscale_height = int(im_original_width/DOWNSCALE_RATIO), int(im_original_height/DOWNSCALE_RATIO)

    x_in = tf.keras.layers.Input(shape=(im_downscale_height, im_downscale_width, im_channels))
    y_out = SoccerBotModel()(x_in)
    model = tf.keras.Model(inputs=x_in, outputs=y_out)
    model.summary()

    model.load_weights(PATH_MODEL_IN)
    print(f"Loaded weights from '{PATH_MODEL_IN}'")

    quant_converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quant_model = quant_converter.convert()

    with open(PATH_MODEL_OUT, "wb+") as fp:
        fp.write(quant_model)
        print(f"Saved model to '{PATH_MODEL_OUT}'")