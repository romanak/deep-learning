import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import tensorflow_datasets as tfds
import time
from tensorboard import program

# Define the constants
BATCH_SIZE = 32
EPOCHS = 100

# Import the orchid dataset
builder = tfds.folder_dataset.ImageFolder('orchid/')
NUM_CLASSES = builder.info.features["label"].num_classes
# print(builder.info)
train_ds = builder.as_dataset(split='train', shuffle_files=True)
valid_ds = builder.as_dataset(split='valid', shuffle_files=True)

# Normalize and resize the dataset
def process_image(example):
    image, label = example['image'], example['label']
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images to 224x224
    image = tf.image.resize(image, (224,224))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
print("Training data size:", train_ds_size)
print("Validation data size:", valid_ds_size)

# Shuffle the dataset and prepare batch generator

train_ds = (train_ds
                  .map(process_image)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

valid_ds = (valid_ds
                  .map(process_image)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

def create_densenet_model():
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

def create_resnet_model():
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

def visualize(history, txt):
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

for i in range(3):
    tic = time.perf_counter()
    name = f'resnet_{i}_{NUM_CLASSES}c_{EPOCHS}e'
    
    # Log training data for tensorboard
    root_logdir = os.path.join(os.curdir, f"log/{name}/")
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', root_logdir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = name + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch')
    
    model = create_resnet_model()
    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(train_ds,
          epochs=EPOCHS,
          validation_data=valid_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb, cp_callback])
    model.save(name + '.h5')
    toc = time.perf_counter()
    txt = f"{name} ({toc - tic:0.1f} s)"
    visualize(history, txt)

for i in range(3):
    tic = time.perf_counter()
    name = f'densenet_{i}_{NUM_CLASSES}c_{EPOCHS}e'
    
    # Log training data for tensorboard
    root_logdir = os.path.join(os.curdir, f"log/{name}/")
    def get_run_logdir():
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', root_logdir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = name + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch')

    model = create_densenet_model()
    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(train_ds,
          epochs=EPOCHS,
          validation_data=valid_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb, cp_callback])
    model.save(name + '.h5')
    toc = time.perf_counter()
    txt = f"{name} ({toc - tic:0.1f} s)"
    visualize(history, txt)