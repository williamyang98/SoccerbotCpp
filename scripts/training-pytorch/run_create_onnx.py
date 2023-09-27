if __name__ == '__main__':
    import argparse
    from models.select_model import get_model_types, select_model
    MODEL_TYPES = get_model_types()
    DEFAULT_MODEL_TYPE = MODEL_TYPES[0]
    DEFAULT_MODEL_PATH = "./data/checkpoint-*.pt"
    DEFAULT_ONNX_PATH = "./data/onnx-*.onnx"

    parser = argparse.ArgumentParser(description="Convert pytorch model to onnx", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES, help="Type of model")
    parser.add_argument("--model-in", type=str, default=DEFAULT_MODEL_PATH, help="Input path for trained model. * is replaced with --model-type.")
    parser.add_argument("--model-out", type=str, default=DEFAULT_ONNX_PATH, help="Output path for onnx model. * is replaced with --model-type.")
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    parser.add_argument("--device", type=str, default="directml", help="Device used by checkpoint. Use 'CPU' for cpu training.")
    args = parser.parse_args()

    # get the generator config
    import sys
    sys.path.append("../")
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
    
    # device which checkpoint is stored as
    import torch
    if args.device == "directml":
        import torch_directml
        DEVICE = torch_directml.device()
    else:
        DEVICE = torch.device(args.device)

    # create the model
    import torchsummary
    SoccerBotModel = select_model(args.model_type)
    DOWNSCALE_RATIO = SoccerBotModel.DOWNSCALE_RATIO
    im_downscale_width, im_downscale_height = int(im_original_width/DOWNSCALE_RATIO), int(im_original_height/DOWNSCALE_RATIO)
    model = SoccerBotModel()
    torchsummary.summary(model, (im_downscale_width, im_downscale_height, im_channels))
    model = model.to(DEVICE)
    
    if os.path.exists(PATH_MODEL_IN):
        try:
            checkpoint = torch.load(PATH_MODEL_IN)
            model.load_state_dict(checkpoint['model_state_dict'])
            curr_epoch = checkpoint.get('curr_epoch', 0)
            average_loss = checkpoint.get('average_loss', torch.inf)
            print(f"Checkpoint loaded from '{PATH_MODEL_IN}' with epoch={curr_epoch}, loss={average_loss:.3e}")
        except Exception as ex:
            print(f"Checkpoint failed to load from '{PATH_MODEL_IN}': {ex}")
            exit(1)
    else:
        print(f"Checkpoint wasn't found at '{PATH_MODEL_IN}'")
        exit(1)

    onnx_model = model.cpu()
    onnx_model.eval()
    x_in = torch.randn(1, im_downscale_height, im_downscale_width, im_channels, requires_grad=True)
    torch.onnx.export(onnx_model, x_in, PATH_MODEL_OUT)
    print(f"Output onnx model to: '{PATH_MODEL_OUT}'")
