from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, Reshape, UpSampling2D, Activation


def model():
    vgg = VGG16(include_top=False, input_shape=(128, 128, 3))
    x = vgg.output

    y = x
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    probability = Dense(5, activation='sigmoid', name='prob_output')(x)

    y = UpSampling2D((3, 3))(y)
    y = Activation('relu')(y)
    y = Conv2D(1, (3, 3), activation='linear')(y)
    position = Reshape(target_shape=(10, 10), name='pos_output')(y)
    return Model(inputs=vgg.input, outputs=[probability, position])


if __name__ == '__main__':
    network = model()
    network.summary()
