from tensorflow.python.keras.callbacks import Callback
import numpy as np
from tensorflow.python.keras import backend as K


class CustomSchedule(Callback):
    def __init__(self, num_hidden, warmup_steps=16000):
        self.num_hidden = num_hidden
        self.warmup_steps = warmup_steps
        self.curr_step = 1

    def schedule(self):
        arg1 = 1 / np.sqrt(self.curr_step)
        arg2 = self.curr_step * (self.warmup_steps ** -1.5)
        self.curr_step += 1
        return 1 / np.sqrt(self.num_hidden) * np.minimum(arg1, arg2)

    def on_train_batch_begin(self, batch, logs=None):
        K.set_value(self.model.optimizer.lr, self.schedule())