# first neural network with keras tutorial
import numpy as np
import random
from keras.layers import Dense, Input
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras





class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))





class Learner:

    def __init__(self, buffer):

        # set buffer
        self.buffer = buffer
        # creating model
        inputs = Input(shape=(4,))
        dense1 = Dense(128, activation='relu')(inputs)
        dense2 = Dense(128, activation='relu')(dense1)
        #dense3 = Dense(128, activation='relu')(dense2)

        # create classification output
        classification_output = Dense(81, activation='softmax')(dense2)

        self.model = Model(inputs, classification_output)
        self.model.summary()

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.history = LossHistory()
        self.data_dump = []

        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    #def learn_step(self, x_data, y_data):
    def learn_step(self):

        batch = self.buffer.sample(32)

        self.model.fit(batch[0], batch[1], callbacks=[self.history], batch_size=32)
        # summarize history for accuracy
        if len(self.history.losses) > 0:
            self.data_dump.append(self.history.losses[0])


    def plot_graph(self):
        #print("save loss")
        plt.plot(range(len(self.data_dump)), self.data_dump)
        plt.title('model losses')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('books_read.png')


    def evaluate(self, x_data, y_data):
        _, accuracy = self.model.evaluate(x_data, y_data)
        print('Accuracy: %.2f' % (accuracy*100))


    def predict(self, x_data):
        return self.model.predict(x_data)


    def save_model(self):
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        print("Saved model to disk")


    def load_model(self):
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights("model.h5")
        print("Loaded model from disk")








class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, inp, label):
        data = (inp, label)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        inp_t, label_t = [], []
        for i in idxes:
            data = self._storage[i]
            inp, label = data
            inp_t.append(np.array(inp, copy=False))
            label_t.append(np.array(label, copy=False))
        return np.array(inp_t), np.array(label_t)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)