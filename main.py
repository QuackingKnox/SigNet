import pandas as pd
import os
import shutil
from itertools import product

import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Layer, BatchNormalization, ZeroPadding2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

SEED = 1337

root = '/content/'
master_zip = '/content/drive/MyDrive/Signatures/Master dataset.zip'
shutil.copy(master_zip, root+'my_zip.zip')
shutil.unpack_archive(root+'my_zip.zip', root)
df = pd.read_csv(root+'Master dataset/4datasets.csv')
folder_path = root+'Master dataset/dataset' 

df['img1_path'] = df['img1'].apply(lambda x: folder_path + '/' + x)
df['img2_path'] = df['img2'].apply(lambda x: folder_path + '/' + x)
df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)

def augment_signature(img, seed):
    img = tf.image.random_flip_left_right(img, seed=seed)
    img = tf.image.random_flip_up_down(img, seed=seed)
    img = tf.image.random_brightness(img, max_delta=0.2, seed=seed)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2, seed=seed)
    return img

def preprocess_signature(path, img_size=(170, 242), augment=False, seed=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=1)
    img = tf.image.resize(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
    img = 1 - img / 255.0
    if augment and seed is not None:
        img = augment_signature(img, seed)
    return img

def process_path(img1_path, img2_path, label, augment=False):
    seed = randint(0, 100000)
    img1 = preprocess_signature(img1_path, augment=augment, seed=seed)
    img2 = preprocess_signature(img2_path, augment=augment, seed=seed)
    return (img1, img2), label

def create_tf_dataset(data, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((data['img1_path'], data['img2_path'], data['label']))
    dataset = dataset.map(lambda img1_path, img2_path, label: process_path(img1_path, img2_path, label, augment=augment), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=batch_size*4, reshuffle_each_iteration=True, seed=SEED).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = tf.cast(y_true, y_pred.dtype)
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def cosine_distance(vecs):
    y_t, y_p = vecs
    y_t = K.l2_normalize(y_t, axis=-1)
    y_p = K.l2_normalize(y_p, axis=-1)
    similarity = K.sum(y_t * y_p, axis=1, keepdims=True)
    distance = 1 - similarity
    return K.expand_dims(K.mean(distance, axis=1), axis=-1)
    
def create_base_network_signet(input_shape):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=1, padding='same', activation='relu', input_shape=input_shape, kernel_initializer='glorot_uniform'))
    model.add(Lambda(tf.nn.local_response_normalization))
    model.add(MaxPooling2D((3,3), strides=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Lambda(tf.nn.local_response_normalization))
    model.add(MaxPooling2D((3,3), strides=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    
    model.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu', kernel_initializer='glorot_uniform'))
    model.add(MaxPooling2D((3,3), strides=(2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.0005), activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.0005), activation='relu', kernel_initializer='glorot_uniform')) 

    return model
    
    
def compute_accuracy_threshold(labels, predictions):
    max_acc = 0
    best_thresh = -1
    predictions = predictions.ravel()
    labels = labels.ravel()
    thresholds = np.unique(predictions)
    for threshold in thresholds:
        y_pred_class = (predictions < threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, y_pred_class).ravel()
        accuracy = (tp + tn) / (tp + fp + fn + tn)  # Здесь можно посчитать и другие метрики

        if (accuracy > max_acc):
            max_acc, best_thresh = accuracy, threshold


    return max_acc, best_thresh

input_shape = (170, 242, 1)
base_network = create_base_network_signet(input_shape)

img_a = Input(shape=input_shape)
img_b = Input(shape=input_shape)

encoded_a = base_network(img_a)
encoded_b = base_network(img_b)

distance = Lambda(cosine_distance)([encoded_a, encoded_b])

siamese_net = Model(inputs=[img_a, img_b], outputs=distance)


train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=SEED)
test_df, val_df = train_test_split(test_val_df, test_size=0.4, random_state=SEED)

batch_size = 128

train_dataset = create_tf_dataset(train_df, batch_size=batch_size, augment=True)
test_dataset = create_tf_dataset(test_df, batch_size=batch_size, augment=True)
val_dataset = create_tf_dataset(val_df, batch_size=batch_size, augment=False)


rms = tf.keras.optimizers.RMSprop(learning_rate=1e-4, rho=0.9, epsilon=1e-08)
siamese_net.compile(loss=contrastive_loss, optimizer=rms)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('/content/signet-{epoch:03d}.h5', verbose=0, save_weights_only=True)
]

history = siamese_net.fit(train_dataset,
                          validation_data=test_dataset,
                          epochs=20,
                          callbacks=callbacks)
                          
