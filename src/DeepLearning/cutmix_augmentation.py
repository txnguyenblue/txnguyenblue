import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import argparse
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow import keras
from config import CONFIG

sys.path.insert(1, str(CONFIG.utils))
np.random.seed(42)
tf.random.set_seed(42)

from logger import LOGGER


#======= Data Preprocssing =======================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

LOGGER.info(f"Current train dataset: {x_train.shape}, {y_train.shape}")
LOGGER.info(f"Current test dataset: {x_test.shape}, {y_test.shape}")

class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMG_SIZE = 32

def preprocess_image(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.convert_image_dtype(image, tf.float32) / 255.0
    return image, label

train_ds_one = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)

train_ds_two = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .shuffle(1024)
    .map(preprocess_image, num_parallel_calls=AUTO)
)

train_ds_simple = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_ds_simple = (
    train_ds_simple.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
test_ds = (
    test_ds.map(preprocess_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

#========================== END =========================================================

#=========================== HELPERS FUNCTION ===========================================

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)
    cut_w = IMG_SIZE * cut_rat #rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_SIZE * cut_rat
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)

    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1
    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1
    return boundaryx1, boundaryy1, target_h, target_w

@tf.function
def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    lambda_value = sample_beta_distribution(1, alpha, beta)
    lambda_value = lambda_value[0][0]

    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )

    image2 = tf.image.crop_to_bounding_box(
        crop2, boundaryy1, boundaryx1, target_h, target_w
    )

    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )

    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryx1, boundaryy1, target_h, target_w
    )

    image1 = image1 - img1
    image = image1 + image2
    
    lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
    lambda_value = tf.cast(lambda_value, tf.float32)

    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label
#======================= END =============================================================

#======================= MAIN LOGIC ======================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Implementing cutmix augmentation technique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--type", default="train",
                        help="""Whether to run the script in test or train mode. If 'test' 
                        mode, cutmix images will be showvased""")
    return parser.parse_args()

def main(args):
    if args.type == "test":
        train_ds_cmu = (
            train_ds.shuffle(1024)
            .map(cutmix, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )

        image_batch, label_batch = next(iter(train_ds_cmu))
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.title(class_names[np.argmax(label_batch[i])])
            plt.imshow(image_batch[i])
            plt.axis("off")
    else:
        pass

#======================= END =============================================================

if __name__ == "__main__":
    args = parse_args()
    main(args)
