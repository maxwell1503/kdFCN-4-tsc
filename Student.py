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
import re
from utils.constants import ITERATIONS_STUDENT, EPOCHS, LAYERS, SEPARABLE_CONV


class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

# Create the student
def create_Student(x_train, y_train, x_test, y_test, input_shape, nb_classes, output_directory, filters, filters2, alpha, temperature, layers=3, separable_conv=False):
  recupereTeacherLossAccurayTest2=[]

  teacher_recupere_model = re.sub('results_(.*)Student', 'teacher', output_directory)
  teacher_recupere_model = re.sub('alpha\d+(\.|)\d*', '', teacher_recupere_model)
  teacher_recupere_model = re.sub('temperature\d+(\.|)\d*', '', teacher_recupere_model)

  print(teacher_recupere_model)

  teacher = keras.models.load_model(f"{teacher_recupere_model}/best_model_teacher.h5")
  
  for i in range (ITERATIONS_STUDENT):

    inputs = tf.keras.layers.Input(input_shape)
    if separable_conv:
        conv1 = tf.keras.layers.SeparableConv1D(filters=filters, kernel_size=8, padding="same")(inputs)
    else:
        conv1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=8, padding="same")(inputs)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)

    if layers>1:
        if separable_conv:
            conv2 = tf.keras.layers.SeparableConv1D(filters=filters2, kernel_size=5, padding="same")(conv1)
        else:
            conv2 = tf.keras.layers.Conv1D(filters=filters2, kernel_size=5, padding="same")(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)
        if layers>2:
            if separable_conv:
                conv3 = tf.keras.layers.SeparableConv1D(filters=filters, kernel_size=3, padding="same")(conv2)
            else:
                conv3 = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding="same")(conv2)
            conv3 = tf.keras.layers.BatchNormalization()(conv3)
            conv3 = tf.keras.layers.ReLU()(conv3)
            gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        else:
            gap = tf.keras.layers.GlobalAveragePooling1D()(conv2)

    else:
        gap = tf.keras.layers.GlobalAveragePooling1D()(conv1)

    outputs = tf.keras.layers.Dense(nb_classes)(gap)

    student = keras.Model(inputs=inputs, outputs=outputs, name="student")
    
    callbacks =[
        keras.callbacks.ModelCheckpoint(
            f"{output_directory}/best_model_distiller{i}.tf", save_weights_only=True, monitor="student_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="student_loss", factor=0.5, patience=50, min_lr=0.0001
        ),
    ]

    # Initialize and compile distiller
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.CategoricalAccuracy()],
        student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=2,
    )

    # Distill teacher to student
    batch_size = 16    
    mini_batch_size = int(min(x_train.shape[0]/10, batch_size))   
    
    history2= distiller.fit(x_train, y_train,batch_size=mini_batch_size, epochs=EPOCHS,validation_data=(x_test,y_test),
          callbacks=callbacks,)
    # Evaluate student on test dataset
    resultat = distiller.evaluate(x_test, y_test)
    recupereTeacherLossAccurayTest2.append(resultat)
    histo_df = pd.DataFrame(history2.history)
    hist_csv_file = output_directory +'/historyfold'+str(i)+'.csv'
    with open(hist_csv_file,mode='w' ) as f:
      histo_df.to_csv(f)

    
    loss =history2.history['student_loss'] 
    val_loss =history2.history['val_student_loss']
    tech=recupereTeacherLossAccurayTest2
    np.savetxt(output_directory + 'tech.out',tech,delimiter=',')