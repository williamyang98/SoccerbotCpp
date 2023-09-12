import threading
from collections import deque

# generate data in separate thread, load from main thread
class AsyncBatchGenerator:
    def __init__(self, batch_generator, max_batches=2):
        self.max_batches = max_batches
        self.batches = deque([])
        self.batches_cv = threading.Condition()

        self.batch_generator = batch_generator
        self.generator_thread = threading.Thread(target=lambda: self._generate_data())
        self.is_running = False
        self.start()

    def __del__(self):
        self.stop()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.generator_thread.start()

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        with self.batches_cv:
            self.batches_cv.notify_all()
        self.generator_thread.join()

    def _generate_data(self):
        while self.is_running:
            with self.batches_cv:
                while len(self.batches) >= self.max_batches:
                    if not self.is_running:
                        return
                    self.batches_cv.wait()
                batch = self.batch_generator()
                self.batches.append(batch)
                self.batches_cv.notify()

    def get_batch(self):
        with self.batches_cv:
            while len(self.batches) == 0:
                self.batches_cv.wait()   
            batch = self.batches.popleft()
            self.batches_cv.notify()
        return batch