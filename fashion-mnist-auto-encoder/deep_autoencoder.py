from keras.layers import Flatten, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MSE
from keras.datasets import fashion_mnist
import numpy as np
import keras
import matplotlib.pyplot as plt


# load data set
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# normalizaton images pixels
train_x = train_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.

# flat images
train_x = train_x.reshape((len(train_x), np.prod(train_x.shape[1:])))
test_x = test_x.reshape((len(test_x), np.prod(test_x.shape[1:])))


# define encoder layers as functional keras api
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

# define decoder layers as functional keras api
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# combine encoder and decoder together to create autoencoder.
autoencoder = Model(input_img, decoded)

# compile model to set optimizer and loss function
autoencoder.compile(optimizer=Adam(), loss=MSE)

autoencoder.summary()

"""**A custom callback to show random images at end of epoch and show error amount.**"""

class ShowResult(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            "The train loss for epoch {} is {:7.4f} and validation loss is {:7.4f}.".format(
                epoch, logs["loss"], logs["val_loss"]
            )
        )

        # select random images.
        random_image_1 = np.random.randint(0, len(self.validation_data[0])-1)
        random_image_2 = np.random.randint(0, len(self.validation_data[0])-1)
        random_image_1 = self.validation_data[0][random_image_1]
        random_image_2 = self.validation_data[0][random_image_2]
        # plot images
        f = plt.figure()
        f.add_subplot(2,2, 1)
        predict = self.model.predict(np.array([random_image_1]))
        plt.title('predicted')
        plt.imshow(predict.reshape((28, 28)))
        plt.axis('off')
        f.add_subplot(2,2, 2)
        plt.title('original')
        plt.imshow(random_image_1.reshape((28, 28)))
        plt.axis('off')
        f.add_subplot(2,2, 3)
        predict = self.model.predict(np.array([random_image_2]))
        plt.title('predicted')
        plt.imshow(predict.reshape((28, 28)))
        plt.axis('off')
        f.add_subplot(2,2, 4)
        plt.title('original')
        plt.imshow(random_image_2.reshape((28, 28)))
        plt.axis('off')
        plt.show()

# start training
autoencoder.fit(train_x, train_x,
                epochs=30,
                batch_size=256,
                verbose=0,
                shuffle=True, # each epoch training data are shuffled.
                validation_data=(test_x, test_x),
                callbacks=[ShowResult()]
)