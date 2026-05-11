import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, InputLayer, Dropout, Conv1D, Conv2D,
    Flatten, Reshape, MaxPooling1D, MaxPooling2D, AveragePooling2D,
    BatchNormalization, Permute, ReLU, Softmax, Activation)
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

sa = SpecAugment(
    spectrogram_shape=[int(input_length / 13), 13],
    mF_num_freq_masks=1,             
    F_freq_mask_max_consecutive=2,    
    mT_num_time_masks=1,
    T_time_mask_max_consecutive=2,    
    enable_time_warp=False,
    W_time_warp_max_distance=6,
    mask_with_mean=False
)
train_dataset = train_dataset.map(sa.mapper(), num_parallel_calls=tf.data.AUTOTUNE)

EPOCHS = args.epochs or 50               
LEARNING_RATE = args.learning_rate or 0.001  
ENSURE_DETERMINISM = args.ensure_determinism
BATCH_SIZE = args.batch_size or 32

if not ENSURE_DETERMINISM:
    train_dataset = train_dataset.shuffle(buffer_size=BATCH_SIZE * 4)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False)
validation_dataset = validation_dataset.batch(BATCH_SIZE, drop_remainder=False)

model = Sequential()

model.add(tf.keras.layers.GaussianNoise(stddev=0.1, input_shape=(input_length,)))

model.add(Reshape((int(input_length / 13), 13)))

model.add(Conv1D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.25))

model.add(Conv1D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())


model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(classes, name='y_pred', activation='softmax'))

opt = Adam(learning_rate=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

callbacks.append(
    BatchLoggerCallback(
        BATCH_SIZE,
        train_sample_count,
        epochs=EPOCHS,
        ensure_determinism=ENSURE_DETERMINISM
    )
)

callbacks.append(
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    verbose=2,
    callbacks=callbacks
)

disable_per_channel_quantization = False