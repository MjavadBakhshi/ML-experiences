from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.datasets import fashion_mnist
import numpy as np
import keras
import matplotlib.pyplot as plt

# load data set
(train_x, train_y), (test_x, test_y) = fashion_mnist.load_data()

# normalizaton images pixels
train_x = train_x.astype('float32') / 255.
test_x = test_x.astype('float32') / 255.

# reshape to 28*28*1
train_x = train_x.reshape((len(train_x), 28, 28, 1))
test_x = test_x.reshape((len(test_x), 28, 28, 1))


# define encoder layers as functional keras api
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# define decoder layers as functional keras api
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
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
        plt.title('preidcted')
        plt.imshow(predict.reshape((28, 28)))
        plt.axis('off')
      
        f.add_subplot(2,2, 2)
        
        plt.title('original')
        plt.imshow(random_image_1.reshape((28, 28)))
        plt.axis('off')
        f.add_subplot(2,2, 3)
        predict = self.model.predict(np.array([random_image_2]))
        plt.title('preidcted')
        plt.imshow(predict.reshape((28, 28)))
        plt.axis('off')
        f.add_subplot(2,2, 4)
        plt.title('original')
        plt.imshow(random_image_2.reshape((28, 28)))
        plt.axis('off')
        plt.show()

# start training
autoencoder.fit(train_x, train_x,
                epochs=15,
                batch_size=256,
                verbose=0,
                shuffle=True, # each epoch training data are shuffled.
                validation_data=(test_x, test_x),
                callbacks=[ShowResult()]
)