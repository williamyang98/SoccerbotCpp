if __name__ == "__main__":
    import argparse
    DEFAULT_MODEL_PATH = "./models/model_train.h5f"

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    parser.add_argument("--model-in", type=str, default=DEFAULT_MODEL_PATH, help="Input path for pretrained model")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_PATH, help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train model")
    parser.add_argument("--device", type=str, default="GPU:0", choices=["CPU", "GPU:0"], help="Device to use for training")
    args = parser.parse_args()

    import os
    import pathlib

    if not os.path.exists(args.asset_path):
        print(f"[error] invalid asset path: '{args.asset_path}'")
        exit(1)

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

    def create_batch(batch_size):
        X_in = np.zeros((batch_size, im_height, im_width, im_channels), dtype=np.float32)
        Y_out = np.zeros((batch_size, 3), dtype=np.float32)

        for i in range(batch_size):
            image, bounding_box, has_ball = generator.create_sample()
            image = image.convert("RGB")
            image = image.resize((int(x/IMAGE_DOWNSCALE) for x in image.size))
            
            x_in = np.asarray(image)
            x_in = x_in.astype(np.float32) / 255.0
            x_in = x_in[np.newaxis,:]
            X_in[i] = x_in

            x_center, y_center, ball_width, ball_height = bounding_box
            confidence = 0.0 if not has_ball else 1.0

            Y_out[i,0] = x_center
            Y_out[i,1] = y_center
            Y_out[i,2] = confidence

        X_in = tf.convert_to_tensor(X_in)
        Y_out = tf.convert_to_tensor(Y_out)
        return (X_in, Y_out)

    from async_batch_generator import AsyncBatchGenerator 

    import tensorflow as tf
    def create_model():
        from model import SoccerBotModel
        module = SoccerBotModel()
        x_in = tf.keras.layers.Input(shape=(im_height, im_width, im_channels))
        y_out = module(x_in)
        model = tf.keras.Model(inputs=x_in, outputs=y_out)
        return model
    
    def calculate_metrics(Y_pred, Y_expected):
        # shape = B,3 => x_center, y_center, confidence
        
        # accuracy of confidence score
        threshold = 0.5
        pred_confidence = tf.cast(tf.math.greater(Y_pred[:,2], threshold), tf.float32)
        expected_confidence = Y_expected[:,2]
        confidence_accuracy = 1.0-tf.math.reduce_mean(tf.math.abs(pred_confidence - expected_confidence))

        # error of confidence score
        mean_confidence_error = tf.math.reduce_mean(tf.math.abs(Y_pred[:,2] - Y_expected[:,2]))

        # bounding box error when there is a ball
        pred_position = Y_pred[:,:2]
        expected_position = Y_expected[:,:2]
        position_error = tf.math.reduce_sum(tf.math.abs(pred_position-expected_position), axis=1)
        mean_position_error = tf.math.reduce_mean(tf.math.multiply(position_error, expected_confidence))

        # anomalous bounding box when there isn't a ball 
        mean_position_anomaly = tf.math.reduce_mean(tf.math.multiply(position_error, 1-expected_confidence))
        
        return tf.stack([confidence_accuracy, mean_confidence_error, mean_position_error, mean_position_anomaly])
    
    model = create_model()
    model.summary()

    # training setup
    hyperparams = {
        "init_lr": 1e-3, 
        "total_train_batches": 20,
        "total_test_batches": 2,
        "train_batch_size": 64,
        "test_batch_size": 64,
        "epochs": args.epochs,
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparams["init_lr"])
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    try:
        model.load_weights(args.model_in)
        print(f"Loaded weights from '{args.model_in}'")
    except Exception as ex:
        print(f"Failed to load in weights from '{args.model_in}': {ex}")
    
    print(f"Start of training")

    async_train_batch_generator = AsyncBatchGenerator(lambda: create_batch(hyperparams["train_batch_size"]), max_batches=2)
    async_test_batch_generator = AsyncBatchGenerator(lambda: create_batch(hyperparams["test_batch_size"]), max_batches=2)
    try:
        curr_epoch = 0
        # NOTE: Run using directml
        with tf.device(f"/device:{args.device}"):
            for curr_epoch in range(hyperparams["epochs"]):
                train_loss_avg = tf.keras.metrics.Mean()
                for _ in range(hyperparams["total_train_batches"]):
                    X_in, Y_out = async_train_batch_generator.get_batch()
                    with tf.GradientTape() as tape:
                        Y_pred = model(X_in, training=True)
                        loss = loss_fn(Y_out, Y_pred)
                        grad = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grad, model.trainable_variables))
                    train_loss_avg.update_state(loss)
            
                test_loss_avg = tf.keras.metrics.Mean()
                test_metrics_avg = tf.keras.metrics.MeanTensor()
                for i in range(hyperparams["total_test_batches"]):
                    X_in, Y_out = async_test_batch_generator.get_batch()
                    Y_pred = model(X_in)
                    loss = loss_fn(Y_out, Y_pred)        
                    metrics = calculate_metrics(Y_pred, Y_out)
            
                    test_loss_avg.update_state(loss)
                    test_metrics_avg.update_state(metrics)
                    
                confidence_accuracy, mean_confidence_error, mean_position_error, mean_position_anomaly = test_metrics_avg.result()
                print(f"epoch={curr_epoch} | train_loss={train_loss_avg.result():.2e} test_loss={test_loss_avg.result():.2e} | " +
                    f"conf_acc={confidence_accuracy:.2e} conf_err={mean_confidence_error:.2e} | " + 
                    f"pos_err={mean_position_error:.2e} pos_anom={mean_position_anomaly:.2e}")
    except KeyboardInterrupt:
        print(f"Interrupted at epoch={curr_epoch}")
    
    model.save_weights(args.model_out)
    print(f"Saved weights to '{args.model_out}'")
    exit(0)

    