# first neural network with keras tutorial
from keras.layers import Dense, Input
from keras.models import Model
from keras.models import model_from_json
from keras.optimizers import Adam
import matplotlib.pyplot as plt
#from keras.callbacks import Callback
import keras

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Learner:

    def __init__(self):

        # creating model
        inputs = Input(shape=(4,))
        dense1 = Dense(256, activation='relu')(inputs)
        dense2 = Dense(128, activation='relu')(dense1)
        #dense3 = Dense(128, activation='relu')(dense2)

        # create classification output
        classification_output = Dense(81, activation='softmax')(dense2)

        self.model = Model(inputs, classification_output)
        self.model.summary()

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.history = LossHistory()
        self.data_dump = []

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def learn_step(self, x_data, y_data):

        self.model.fit(x_data, y_data, callbacks=[self.history])
        #print(self.history.losses)
        # summarize history for accuracy
        if len(self.history.losses) > 0:
            print(self.history.losses[0])
            self.data_dump.append(self.history.losses[0])


    def plot_graph(self):
        print("save loss")
        plt.plot(range(len(self.data_dump)), self.data_dump)
        plt.title('model losses')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig('books_read.png')
        #plt.close()



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
