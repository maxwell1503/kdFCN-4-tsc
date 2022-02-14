import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras.layers import Conv2D,GlobalAveragePooling2D,Dense,Softmax,Flatten,MaxPooling2D,Dropout,Activation, Lambda, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import kullback_leibler_divergence as KLD_Loss, categorical_crossentropy as logloss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import categorical_accuracy
from utils.constants import EPOCHS

# Create the teacher
def create_teacher(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2):
    recupereTeacherLossAccurayTest= []
    
    inputs = tf.keras.layers.Input(input_shape)

    conv1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=8, padding="same")(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=filters2, kernel_size=5, padding="same")(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)

    conv3 = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)

    gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)
    outputs = tf.keras.layers.Dense(nb_classes)(gap)

    teacher = keras.Model(inputs=inputs, outputs=outputs, name="teacher")
    teacher.summary()
    
    # Train teacher as usual
    callbacks = [
    keras.callbacks.ModelCheckpoint(
    f"{output_directory}/best_model_teacher.h5", save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
    monitor="loss", factor=0.5, patience=50, min_lr=0.0001
    ),
      ]
    teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )
    
    # Train and evaluate teacher on data
    batch_size = 16   
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))
    
    history= teacher.fit(x_train, y_train,batch_size=mini_batch_size, epochs=EPOCHS,validation_data=(x_test,y_test),
    callbacks=callbacks,)
    
    resultat = teacher.evaluate(x_test, y_test)
    recupereTeacherLossAccurayTest.append(resultat)
    histo_dfteacher = pd.DataFrame(history.history)
    hist_csv_file = output_directory + '/historyteacher' + '.csv'
    with open(hist_csv_file,mode='w' ) as f:
        histo_dfteacher.to_csv(f)
    loss =history.history['loss'] 
    val_loss =history.history['val_loss']

    
    metric = "categorical_accuracy"
    
    tech=recupereTeacherLossAccurayTest
    np.savetxt(output_directory + 'tech.out',tech,delimiter='\t')      