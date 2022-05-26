import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import time
import zipfile

def get_dataset(ds_name, BATCH_SIZE):
    with zipfile.ZipFile(f"datasets/{ds_name}.zip", 'r') as zip_ref:
        zip_ref.extractall("datasets/")

    # Import the orchid dataset
    builder = tfds.folder_dataset.ImageFolder(f'datasets/{ds_name}/')
    NUM_CLASSES = builder.info.features["label"].num_classes
    # print(builder.info)
    # tfds.show_examples(train_ds, builder.info)
    train_ds = builder.as_dataset(split='train', shuffle_files=True)
    test_ds = builder.as_dataset(split='test', shuffle_files=True)
    valid_ds = builder.as_dataset(split='valid', shuffle_files=True)

    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
    print("Training data size:", train_ds_size)
    print("Test data size:", test_ds_size)
    print("Validation data size:", valid_ds_size)

    # Shuffle the dataset and prepare batch generator

    train_ds = (train_ds
                      .map(process_image)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=BATCH_SIZE, drop_remainder=True))
    test_ds = (test_ds

                      .map(process_image)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=BATCH_SIZE, drop_remainder=True))
    valid_ds = (valid_ds
                      .map(process_image)
                      .shuffle(buffer_size=train_ds_size)
                      .batch(batch_size=BATCH_SIZE, drop_remainder=True))

    return (train_ds, test_ds, valid_ds, NUM_CLASSES)

def create_alexnet_model(NUM_CLASSES):
    """Function to construct AlexNet model"""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(224,224,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])
    # Display the model's architecture
    # model.summary()
    return model

def create_densenet_model(NUM_CLASSES):
    """Function to construct DenseNet model"""
    model = tf.keras.applications.DenseNet169(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=NUM_CLASSES
    )

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

def create_resnet_model(NUM_CLASSES):
    """Function to construct ResNet model"""
    model = tf.keras.applications.resnet.ResNet152(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=NUM_CLASSES
    )

    model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
                loss=tf.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

    return model

def visualize(history, txt, name):
    # summarize history for accuracy
    plt.plot(history.history['sparse_categorical_accuracy'])
    plt.plot(history.history['val_sparse_categorical_accuracy'])
    plt.title('Accuracy for ' + txt)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(name + '_accuracy.png')
    # plt.show()
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss for ' + txt)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(name + '_loss.png')
    # plt.show()
    plt.clf()

# Normalize and resize the dataset
def process_image(example):
    image, label = example['image'], example['label']
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images to 224x224
    image = tf.image.resize(image, (224,224))
    return image, label

def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
