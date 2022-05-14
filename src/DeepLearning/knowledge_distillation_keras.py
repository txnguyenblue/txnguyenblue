import tensorflow as tf
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from config import CONFIG
sys.path.insert(1, str(CONFIG.src))
sys.path.insert(2, str(CONFIG.utils))

from utilities import \
(get_accuracy, visualize_losses, BaseModel, Matrix, Vector, get_numpy_instance)

import logging
from logger import LOGGER

file_dir = Path(__file__).resolve()
file_name = file_dir.stem
class Distiller(keras.Model):

    def __init__(self, teacher, student) -> None:
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn,
                    distillation_loss_fn, alpha: float = 0.1, temperature: int = 3) -> None:
        """Configure the distiller

        Args:
            optimizer (_type_): Keras optimzer for the student weights
            metrics (_type_): Keras metrics for evaluation
            student_loss_fn (_type_): Loss function of difference between student predictions and ground-truth
            distillation_loss_fn (_type_): Loss function of difference between soft student predictions and soft teacher predictions
            alpha (float, optional): weight to student_loss_fn and 1 - alpha to distillation_loss_fn. Defaults to 0.1.
            temperature (int, optional): Temperature for softening probability distributions. Larger temperature gives softer distributions. Defaults to 3.
        """

        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        #Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training = True)
            
            # Compute loss
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1)
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compiled()`
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss, "distillation_loss": distillation_loss
        })

        return results

    def test_step(self, data):
        # Unpack data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performacne
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

    
teacher = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
    layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
    layers.Flatten(),
    layers.Dense(10),
], name="teacher",
)

student = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
    layers.LeakyReLU(alpha=0.2),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
    layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
    layers.Flatten(),
    layers.Dense(10),
    
], name="student"
)


def main():
    LOGGER.info("Load and prepare data...")
    student_scratch = keras.models.clone_model(student)

    # Prepare the dataset
    batch_size = 64
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalize data
    x_train = x_train.astype("float32") / 255.0
    x_train = np.reshape(x_train, (-1, 28, 28, 1))

    x_test = x_test.astype("float32") / 255.0
    x_test = np.reshape(x_test, (-1, 28, 28, 1))

    LOGGER.info("Train teacher...")
    teacher.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    teacher_history = teacher.fit(x_train, y_train, epochs=5, validation_split=0.33)
    teacher.evaluate(x_test, y_test)
    LOGGER.info("End train and test teacher model.")
    LOGGER.info("Distill the teacher to student")
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    student_distiller_history = distiller.fit(x_train, y_train, epochs=3, validation_split=0.33)
    distiller.evaluate(x_test, y_test)
    LOGGER.info("End train and test knowledge distillation")
    LOGGER.info("Train student from scratch for comparison")
    student_scratch.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    student_history = student_scratch.fit(x_train, y_train, validation_split=0.33)
    student_scratch.evaluate(x_test, y_test)
    LOGGER.info("End train and test student alone")
    LOGGER.info("Performance plots:")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, constrained_layout=True)
    LOGGER.info(f"Current teacher history: {teacher_history.history}")
    report_path = CONFIG.reports / file_name
    report_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Saving teacher history to: {str(report_path / 'teacher')}")
    with open(str(report_path / "teacherHistory"), "wb") as f:
        pickle.dump(teacher_history.history, f)
    LOGGER.info(f"Current student distill: {student_distiller_history.history}")
    LOGGER.info(f"Saving student history to: {str(report_path / 'student_distill')}")
    with open(str(report_path / "student_distill"), "wb") as f:
        pickle.dump(student_distiller_history.history, f)
    LOGGER.info(f"Current student: {student_history.history}")
    LOGGER.info(f"Saving student history to: {str(report_path / 'student')}")
    with open(str(report_path / "student"), "wb") as f:
        pickle.dump(student_history.history, f)
    ax1.plot(teacher_history.history["loss"], label="teacher")
    ax1.plot(student_distiller_history.history["student_loss"], label="student distill")
    ax1.plot(student_history.history["loss"], label="student")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Model Loss")
    # ax2.legend(["teacher", "student distilled", "student"], loc="upper left")
    ax2.plot(teacher_history.history["sparse_categorical_accuracy"])
    ax2.plot(student_distiller_history.history["sparse_categorical_accuracy"])
    ax2.plot(student_history.history["sparse_categorical_accuracy"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Training Accuracy")
    ax4.plot(teacher_history.history["val_loss"], label="teacher")
    ax4.plot(student_distiller_history.history["val_student_loss"], label="student distill")
    ax4.plot(student_history.history["val_loss"], label="student")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Val Model Loss")
    ax3.plot(teacher_history.history["val_sparse_categorical_accuracy"], label="teacher")
    ax3.plot(student_distiller_history.history["val_sparse_categorical_accuracy"], label="student distill")
    ax3.plot(student_history.history["val_sparse_categorical_accuracy"], label="student")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Val Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
