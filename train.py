from math import ceil
import tensorflow as tf
from net.network import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from generator import train_generator, valid_generator


def loss_function_1(y_true, y_pred):
    """ Probabilistic output loss """
    a = tf.clip_by_value(y_pred, 1e-20, 1)
    b = tf.clip_by_value(tf.subtract(1.0, y_pred), 1e-20, 1)
    cross_entropy = - tf.multiply(y_true, tf.math.log(a)) - tf.multiply(tf.subtract(1.0, y_true), tf.math.log(b))
    cross_entropy = tf.reduce_mean(cross_entropy, 0)
    loss = tf.reduce_mean(cross_entropy)
    return loss


def loss_function_2(y_true, y_pred):
    """ Positional output loss """
    square_diff = tf.math.squared_difference(y_true, y_pred)
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    square_diff = tf.multiply(square_diff, mask)
    square_diff = tf.reduce_mean(square_diff, 1)
    square_diff = tf.reduce_mean(square_diff, 0)
    loss = tf.reduce_mean(square_diff)
    return loss


# Creating the model
model = model()
model.summary()

# Compile
adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-10, decay=0.0)
loss_function = {"prob_output": loss_function_1, "pos_output": loss_function_2}
model.compile(optimizer=adam, loss=loss_function, metrics=None)

# Train
epochs = 10
batch_size = 256
train_set_size = 25090
valid_set_size = 1317
training_steps_per_epoch = ceil(train_set_size / batch_size)
validation_steps_per_epoch = ceil(valid_set_size / batch_size)

train_gen = train_generator(batch_size=batch_size)
val_gen = valid_generator(batch_size=batch_size)

checkpoints = ModelCheckpoint('weights/weights_{epoch:03d}.h5', save_weights_only=True, save_freq=1)
history = model.fit(train_gen, steps_per_epoch=training_steps_per_epoch, epochs=epochs, verbose=1,
                    validation_data=val_gen, validation_steps=validation_steps_per_epoch,
                    callbacks=[checkpoints], shuffle=True, max_queue_size=512)

with open('weights/history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
