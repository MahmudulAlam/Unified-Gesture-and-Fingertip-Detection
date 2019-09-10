import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from SOLO.model import model
from SOLO.generator import train_generator

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# create model
model = model()
model.summary()
model.load_weights('../weights/solo.h5')

# compile
adam = Adam(lr=1e-6)
model.compile(optimizer=adam, loss='binary_crossentropy')

# train
epochs = 1
train_gen = train_generator(steps_per_epoch=10159, sample_per_batch=4)
checkpoints = ModelCheckpoint('../weights/hand_weights{epoch:03d}.h5', save_weights_only=True, period=1)
history = model.fit_generator(train_gen, steps_per_epoch=10159, epochs=epochs, verbose=1,
                              shuffle=True, callbacks=[checkpoints], max_queue_size=100)

with open('history.txt', 'a+') as f:
    print(history.history, file=f)

print('All Done!')
