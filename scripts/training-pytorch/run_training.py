if __name__ == "__main__":
    import argparse
    from models.select_model import get_model_types, select_model
    MODEL_TYPES = get_model_types()
    DEFAULT_MODEL_TYPE = MODEL_TYPES[0]
    DEFAULT_MODEL_PATH = "./data/checkpoint-*.pt"

    parser = argparse.ArgumentParser(description="Run model training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--asset-path", type=str, default="../assets/", help="Path to game assets")
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE, choices=MODEL_TYPES, help="Type of model")
    parser.add_argument("--model-in", type=str, default=DEFAULT_MODEL_PATH, help="Input path for pretrained model. * is replaced with --model-type.")
    parser.add_argument("--model-out", type=str, default=DEFAULT_MODEL_PATH, help="Output path for trained model. * is replaced with --model-type.")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs to train model")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Total steps per epochs. If a value of 0 is provided, we use the model's default.")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size. If a value of 0 is provided, we use the model's default.")
    parser.add_argument("--total-parallel-load", type=int, default=0, help="Number of threads to spawn for generating data set. If a value of 0 is provided then we use the number of logical processors.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Set optimizer learning rate.")
    parser.add_argument("--device", type=str, default="directml", help="Device to use for training. Use 'CPU' for cpu training.")
    parser.add_argument("--print-model-summary", action="store_true", help="Print model summary before training.")
    parser.add_argument("--no-autosave", action="store_true", help="Disables saving model checkpoints automatically.")
    parser.add_argument("--restart-optimizer", action="store_true", help="Refreshes the optimizer from specified value.")
    parser.add_argument("--reset-loss", action="store_true", help="Sets the best loss to infinite so we start autosaving new models.")
    parser.add_argument("--disable-lr-anneal", action="store_true", help="Disables learning rate reduction after training stalls.")
    args = parser.parse_args()
    
    # validate paths
    import os
    import sys
    import pathlib
    PATH_MODEL_IN = args.model_in.replace("*", args.model_type)
    PATH_MODEL_OUT = args.model_out.replace("*", args.model_type)
    if not os.path.exists(args.asset_path):
        print(f"[error] invalid asset path: '{args.asset_path}'")
        exit(1)

    TOTAL_DATA_THREADS = args.total_parallel_load
    if TOTAL_DATA_THREADS == 0:
        import multiprocessing
        TOTAL_DATA_THREADS = multiprocessing.cpu_count()
    print(f"Using {TOTAL_DATA_THREADS} threads for data loading")
    pathlib.Path(os.path.dirname(PATH_MODEL_OUT)).mkdir(parents=True, exist_ok=True)
        
    # create generator
    sys.path.append("../")
    from generator import GeneratorConfig, BasicSampleGenerator
    import numpy as np
    import glob
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
    
    # load training device
    import torch
    if args.device == "directml":
        import torch_directml
        DEVICE = torch_directml.device()
    else:
        DEVICE = torch.device(args.device)

    # create parallel data loader
    import threading
    import queue
    import multiprocessing
    import collections
    import PIL
    import random

    class ThreadedBatch:
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y
            self.is_in_use = False
            self.cv_in_use = threading.Condition()
            with self.cv_in_use:
                self.cv_in_use.notify_all()

        def wait_until_unused(self):
            with self.cv_in_use:
                while self.is_in_use:
                    self.cv_in_use.wait()

        def flag_in_use(self):
            self.is_in_use = True
            with self.cv_in_use:
                self.cv_in_use.notify()

        def flag_unused(self):
            self.is_in_use = False
            with self.cv_in_use:
                self.cv_in_use.notify()

    class ParallelBatchGenerator:
        def __init__(self, batch_size, image_size, is_training=True, total_threads=0):
            if total_threads == 0:
                total_threads = multiprocessing.cpu_count()

            self.batch_size = batch_size
            self.image_size = image_size
            self.is_training = is_training
            self._queue = collections.deque([])
            self._cv_queue_length = threading.Condition()
            self._threads = [threading.Thread(target=self._thread_loop, args=(i,)) for i in range(total_threads)]

            self._is_running = True
            for thread in self._threads:
                thread.start()

        def __del__(self):
            self.stop()

        def __exit__(self):
            self.stop()

        def stop(self):
            if not self._is_running:
                return

            print("Terminating data generator")
            self._is_running = False
            while len(self._queue) > 0:
                batch = self._queue.popleft()
                batch.flag_unused()
            for thread in self._threads:
                thread.join()

        def _thread_loop(self, index):
            im_downscale_height, im_downscale_width, im_channels = self.image_size

            # preallocate the memory and reuse it
            X_in = torch.zeros((self.batch_size, im_downscale_height, im_downscale_width, im_channels), dtype=torch.float32)
            Y_out = torch.zeros((self.batch_size, 3))
            batch = ThreadedBatch(X_in, Y_out)

            RESIZE_TYPES = [ PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.NEAREST ]

            while self._is_running:
                batch.wait_until_unused()
                if not self._is_running:
                    print(f"Terminating data generator thread: {index}")
                    break

                for i in range(self.batch_size):
                    image, bounding_box, has_ball = generator.create_sample()
                    x_center, y_center, ball_width, ball_height = bounding_box
                    confidence = 0.0 if not has_ball else 1.0

                    # image processing
                    image = image.convert("RGB")
                    im_width, im_height = image.size
                    is_resized = (im_downscale_height != im_height) or (im_downscale_width != im_width)
                    if is_resized:
                        if self.is_training:
                            image = image.resize((im_downscale_width, im_downscale_height), resample=random.choice(RESIZE_TYPES))
                        else:
                            image = image.resize((im_downscale_width, im_downscale_height), resample=PIL.Image.BICUBIC)

                    # shape = h,w,c
                    x_in = np.asarray(image)
                    x_in = x_in.astype(np.float32) / 255.0

                    if self.is_training:
                        # horizontal flip
                        if random.random() > 0.5:
                            x_in = np.flip(x_in, axis=1) 
                            x_center = 1.0-x_center
                        # vertical flip
                        if random.random() > 0.5:
                            x_in = np.flip(x_in, axis=0)
                            y_center = 1.0-y_center

                    # store data
                    x_in = x_in[np.newaxis,:]
                    # NOTE: We have to copy since training augmentations can produce negative stride
                    #       Which pytorch doesn't allow in its tensors
                    x_in = torch.as_tensor(x_in.copy()) # B,H,W,C
                    X_in[i] = x_in
                    Y_out[i,0] = x_center
                    Y_out[i,1] = y_center
                    Y_out[i,2] = confidence

                batch.flag_in_use()
                self._queue.append(batch)
                with self._cv_queue_length:
                    self._cv_queue_length.notify()

        def get_batch(self, timeout=2):
            with self._cv_queue_length:
                while len(self._queue) == 0:
                    self._cv_queue_length.wait(timeout=timeout)
            return self._queue.popleft()
    
    # create model
    SoccerBotModel = select_model(args.model_type)
    DOWNSCALE_RATIO = SoccerBotModel.DOWNSCALE_RATIO
    im_downscale_width, im_downscale_height = int(im_original_width/DOWNSCALE_RATIO), int(im_original_height/DOWNSCALE_RATIO)
    model = SoccerBotModel()
    if args.print_model_summary:
        import torchsummary
        torchsummary.summary(model, (im_downscale_width, im_downscale_height, im_channels))
    model = model.to(DEVICE)
    
    def calculate_metrics(Y_pred, Y_expected):
        # shape = B,3 => x_center, y_center, confidence
        # accuracy of confidence score
        threshold = 0.5
        pred_confidence = Y_pred[:,2] > threshold
        expected_confidence = Y_expected[:,2] > threshold
        confidence_accuracy = (pred_confidence == expected_confidence).to(torch.float32).mean()
        # error of confidence score
        mean_confidence_error = (Y_pred[:,2] - Y_expected[:,2]).abs().mean()
        # bounding box error when there is a ball
        pred_position = Y_pred[expected_confidence,:2]
        expected_position = Y_expected[expected_confidence,:2]
        if pred_position.shape[0] > 0:
            mean_position_error = torch.abs(pred_position-expected_position).sum(axis=-1).mean()
        else:
            mean_position_error = torch.tensor(torch.nan).to(Y_pred.device)

        return torch.tensor([confidence_accuracy, mean_confidence_error, mean_position_error])

    def custom_loss_fn(Y_pred, Y_expected):
        # shape = B,3 => x_center, y_center, confidence
        # error due to prediction
        conf_err = (Y_pred[:,2] - Y_expected[:,2]).abs().mean()
        # error due to position
        pos_err = (Y_pred[:,:2]-Y_expected[:,:2]).abs().sum(axis=1)
        # ignore the error if there wasn't a ball there
        pos_err = pos_err * Y_expected[:,2]
        pos_err = pos_err.mean()

        loss = conf_err + pos_err
        return loss

    # training setup
    BATCH_SIZE = SoccerBotModel.BATCH_SIZE
    STEPS_PER_EPOCH = SoccerBotModel.STEPS_PER_EPOCH
    if args.batch_size != 0:
        BATCH_SIZE = args.batch_size
    if args.steps_per_epoch != 0:
        STEPS_PER_EPOCH = args.steps_per_epoch

    image_size = (im_downscale_height, im_downscale_width, im_channels)
    dataset = ParallelBatchGenerator(BATCH_SIZE, image_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    curr_epoch = 0
    average_loss = torch.inf
    # temp buffer to store results to calculate average metrics
    Y_train_out = torch.zeros((BATCH_SIZE*STEPS_PER_EPOCH, 3), dtype=torch.float32).to(DEVICE)
    Y_train_pred = torch.zeros((BATCH_SIZE*STEPS_PER_EPOCH, 3), dtype=torch.float32).to(DEVICE)
    
    def save_checkpoint(model, optimizer, curr_epoch, average_loss):
        torch.save({
            'curr_epoch': curr_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'average_loss': average_loss
        }, PATH_MODEL_OUT)
    
    if os.path.exists(PATH_MODEL_IN):
        try:
            checkpoint = torch.load(PATH_MODEL_IN)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint.get('curr_epoch', 0)
            average_loss = checkpoint.get('average_loss', torch.inf)
            print(f"Checkpoint loaded from '{PATH_MODEL_IN}' with epoch={curr_epoch}, loss={average_loss:.3e}")
        except Exception as ex:
            print(f"Checkpoint failed to load from '{PATH_MODEL_IN}': {ex}")
    else:
        print(f"Checkpoint wasn't found at '{PATH_MODEL_IN}'")

    def set_learning_rate(learning_rate):
        for params in optimizer.param_groups:
            params["lr"] = learning_rate
            break

    if args.restart_optimizer:
        set_learning_rate(args.learning_rate)

    if args.reset_loss:
        average_loss = torch.inf

    print(f"Start of training")
    from timeit import default_timer
    import tqdm
    tqdm_bar_format = "{desc}{percentage:3.0f}%|{bar:10}{r_bar}"
    
    IS_AUTOSAVE = not args.no_autosave
    is_save_checkpoint = not IS_AUTOSAVE
    try:
        # run training
        start_epoch = curr_epoch
        end_epoch = start_epoch + args.epochs
        best_average_loss = average_loss

        for curr_epoch in range(start_epoch, end_epoch):
            print(f"Epoch {curr_epoch}/{end_epoch}")

            total_loss = 0.0
            total_loss_samples = 0
            with tqdm.trange(STEPS_PER_EPOCH, bar_format=tqdm_bar_format) as tqdm_range:
                total_samples = 0
                model.train()
                for curr_step in tqdm_range:

                    dt_start = default_timer()
                    batch = dataset.get_batch(timeout=5)
                    dt_end = default_timer()
                    dt_elapsed_get_batch = dt_end-dt_start

                    X_in, Y_out = batch.X, batch.Y
                    X_in = X_in.to(DEVICE)
                    Y_out = Y_out.to(DEVICE)
                    batch.flag_unused()

                    Y_pred = model.forward(X_in)
                    loss = custom_loss_fn(Y_pred, Y_out)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # store output predictions for final average
                    start_sample = total_samples
                    end_sample = total_samples + BATCH_SIZE
                    total_samples = end_sample
                    Y_train_pred[start_sample:end_sample] = Y_pred
                    Y_train_out[start_sample:end_sample] = Y_out
                    if curr_step == STEPS_PER_EPOCH-1:
                        metrics = calculate_metrics(Y_train_pred[:end_sample], Y_train_out[:end_sample])
                    else:
                        metrics = calculate_metrics(Y_pred, Y_out)

                    for param_group in optimizer.param_groups:
                        curr_lr = param_group['lr']
                        break

                    loss = loss.cpu().item()
                    metrics = metrics.cpu().numpy()
                    confidence_accuracy, mean_confidence_error, mean_position_error = metrics

                    total_loss += loss
                    total_loss_samples += 1
                    average_loss = total_loss / total_loss_samples

                    tqdm_range.set_postfix(
                        loss_step=loss, loss_avg=average_loss,
                        conf_acc=confidence_accuracy, conf_err=mean_confidence_error, 
                        pos_err=mean_position_error, 
                        dt_batch=dt_elapsed_get_batch,
                        lr=curr_lr
                    )

            average_loss = total_loss / total_loss_samples
            if not args.disable_lr_anneal:
                scheduler.step(average_loss)

            if IS_AUTOSAVE and average_loss < best_average_loss:
                save_checkpoint(model, optimizer, curr_epoch, average_loss)
                print(f"Loss improved from {best_average_loss:.3e} to {average_loss:.3e}. Autosaving to '{PATH_MODEL_OUT}'")
                best_average_loss = average_loss

    except KeyboardInterrupt:
        print(f"Interrupted training early")
        response = input("Do you want to save weights? [Y/N]: ")
        response = response.lower().strip()
        if not (response == "y" or response == "yes"):
            is_save_checkpoint = False

    if is_save_checkpoint: 
        save_checkpoint(model, optimizer, curr_epoch, average_loss)
        print(f"Saved checkpoint to '{PATH_MODEL_OUT}'")

    dataset.stop()

    