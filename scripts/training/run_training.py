if __name__ == "__main__":
    import argparse
    DEFAULT_MODEL_PATH = "./models/model_train.h5f"

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    parser.add_argument("--model-in", type=str, default=DEFAULT_MODEL_PATH, help="Input path for pretrained model")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_PATH, help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Total epochs to train model")
    parser.add_argument("--steps-per-epoch", type=int, default=20, help="Total steps per epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--total-parallel-load", type=int, default=0, help="Number of threads to spawn for generating data set")
    parser.add_argument("--device", type=str, default="GPU:0", choices=["CPU", "GPU:0"], help="Device to use for training")
    parser.add_argument("--print-model-summary", action="store_true", help="Print model summary before training")
    args = parser.parse_args()

    import os
    import pathlib

    if not os.path.exists(args.asset_path):
        print(f"[error] invalid asset path: '{args.asset_path}'")
        exit(1)

    TOTAL_DATA_THREADS = args.total_parallel_load
    if TOTAL_DATA_THREADS == 0:
        import multiprocessing
        TOTAL_DATA_THREADS = multiprocessing.cpu_count()
    print(f"Using {TOTAL_DATA_THREADS} threads for data loading")

    pathlib.Path(os.path.dirname(args.model_out)).mkdir(parents=True, exist_ok=True)

    from generator import GeneratorConfig, BasicSampleGenerator
    import numpy as np
    import glob

    # load 
    config = GeneratorConfig()
    config.set_background_image(os.path.join(args.asset_path, "icons/blank.png"))
    config.set_ball_image(os.path.join(args.asset_path, "icons/ball_v2.png"))

    emote_filepaths = []
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/success*.png")))
    emote_filepaths.extend(glob.glob(os.path.join(args.asset_path, "icons/emote*.png")))
    config.set_emote_images(emote_filepaths)
    config.set_score_font(os.path.join(args.asset_path, "fonts/segoeuil.ttf"), 92)
    generator = BasicSampleGenerator(config)

    IMAGE_DOWNSCALE = 4
    image, bounding_box, has_ball = generator.create_sample()
    image = image.convert("RGB")
    image = image.resize((int(x/IMAGE_DOWNSCALE) for x in image.size))
    im_width, im_height = image.size
    im_channels = 3

    import tensorflow as tf
    class ArtificialDataset(tf.data.Dataset):
        def _generator():
            while True:
                y_out = np.zeros((3,), dtype=np.float32)

                image, bounding_box, has_ball = generator.create_sample()
                image = image.convert("RGB")
                image = image.resize((int(x/IMAGE_DOWNSCALE) for x in image.size))
                
                x_in = np.asarray(image)
                x_in = x_in.astype(np.float32) / 255.0

                x_center, y_center, ball_width, ball_height = bounding_box
                confidence = 0.0 if not has_ball else 1.0

                x_in = tf.convert_to_tensor(x_in)
                y_out = tf.constant([x_center, y_center, confidence])
                yield (x_in, y_out)

        def __new__(cls):
            output_shape = (im_height,im_width,im_channels)
            return tf.data.Dataset.from_generator(
                cls._generator,
                output_signature=(
                    tf.TensorSpec(shape=output_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=(3,), dtype=tf.float32)
                )
            )
    
    class DatasetAugmentation:
        def __call__(self, image, label, training=True):
            from tensorflow.keras.layers.experimental import preprocessing
            data_augmentation = tf.keras.Sequential([
                preprocessing.RandomContrast(0.3),
            ])
            image = data_augmentation(image, training=training)
            return image, label

    def create_model():
        from model import SoccerBotModel
        module = SoccerBotModel()
        x_in = tf.keras.layers.Input(shape=(im_height, im_width, im_channels))
        y_out = module(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        return model
   
    def detect_accuracy(y_true, y_pred, thresh=0.8):
        true_cls = y_true[:,2]
        pred_cls = y_pred[:,2]
        pred_cls = tf.math.greater(pred_cls, thresh)
        pred_cls = tf.cast(pred_cls, true_cls.dtype)
        
        abs_err = tf.math.abs(true_cls-pred_cls)
        return 1-tf.math.reduce_mean(abs_err)

    def position_accuracy(y_true, y_pred):
        true_cls = y_true[:,2]
        true_pos = y_true[:,:2]
        pred_pos = y_pred[:,:2]
        
        dist_sqr_err = tf.math.square(true_pos-pred_pos)
        dist_sqr_err = tf.math.reduce_sum(dist_sqr_err, axis=1)
        dist_err = tf.math.sqrt(dist_sqr_err)
        
        # only consider when object is there
        dist_err = tf.math.multiply(dist_err, true_cls)
        net_err = tf.math.reduce_sum(dist_err)
        total_objects = tf.math.reduce_sum(true_cls)
        
        mean_err = net_err / total_objects
        return 1-mean_err

    def custom_loss_fn(y_true, y_pred):
        true_cls = y_true[:,2]
        pred_cls = y_pred[:,2]
        true_pos = y_true[:,:2]
        pred_pos = y_pred[:,:2]

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        cat_err = bce(true_cls, pred_cls)
        
        # NOTE: use L1 distance instead of L2, this works better
        pos_err = tf.math.abs(true_pos-pred_pos)
        pos_err = tf.math.reduce_sum(pos_err, axis=1)
        pos_err = tf.math.reduce_mean(pos_err)
        
        W0 = 4.0
        net_err = cat_err + W0*pos_err
        return net_err
        
    
    model = create_model()
    if args.print_model_summary:
        model.summary()

    # training setup
    hyperparams = {
        "init_lr": 1e-3, 
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["init_lr"])

    model.compile(
        loss=custom_loss_fn, 
        optimizer=optimizer,
        metrics=[detect_accuracy, position_accuracy]
    )

    try:
        model.load_weights(args.model_in)
        print(f"Loaded weights from '{args.model_in}'")
    except Exception as ex:
        print(f"Failed to load in weights from '{args.model_in}': {ex}")
    
    ds_augment = DatasetAugmentation()
    dataset = tf.data.Dataset.range(TOTAL_DATA_THREADS)
    dataset = dataset.interleave(
        lambda _: ArtificialDataset(),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(args.batch_size)
    dataset= dataset.map(ds_augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    print(f"Start of training")
    is_save_weights = True
    try:
        with tf.device(f"/device:{args.device}"):
            model.fit(dataset, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs)
    except KeyboardInterrupt:
        print(f"Interrupted training early")
        response = input("Do you want to save weights? [Y/N]: ")
        response = response.lower().strip()
        if not (response == "y" or response == "yes"):
            is_save_weights = False

    if is_save_weights: 
        model.save_weights(args.model_out)
        print(f"Saved weights to '{args.model_out}'")

    