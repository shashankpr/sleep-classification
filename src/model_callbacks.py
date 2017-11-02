import keras.callbacks

class Histories(keras.callbacks.History):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.losses = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class Checkpoints(keras.callbacks.Callback):
    def __init__(self):
        self.checkpointer = None

    def on_train_begin(self, logs=None):
        self.checkpointer = keras.callbacks.ModelCheckpoint(filepath='model/weights.hdf5', monitor='val_loss',
                                                            verbose=1, save_best_only=True)
        return