import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
from tensorboard import program
import zipfile
import utils

# Define the constants
BATCH_SIZE = 32
EPOCHS = 1

with zipfile.ZipFile("datasets/orchid52.zip", 'r') as zip_ref:
    zip_ref.extractall("datasets/")

# Import the orchid dataset
builder = tfds.folder_dataset.ImageFolder('datasets/orchid52/')
NUM_CLASSES = builder.info.features["label"].num_classes
# print(builder.info)
train_ds = builder.as_dataset(split='train', shuffle_files=True)
valid_ds = builder.as_dataset(split='valid', shuffle_files=True)

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
valid_ds_size = tf.data.experimental.cardinality(valid_ds).numpy()
print("Training data size:", train_ds_size)
print("Validation data size:", valid_ds_size)

# Shuffle the dataset and prepare batch generator

train_ds = (train_ds
                  .map(utils.process_image)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

valid_ds = (valid_ds
                  .map(utils.process_image)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=BATCH_SIZE, drop_remainder=True))

i=0
tic = time.perf_counter()
name = f'densenet_{i}_{NUM_CLASSES}c_{EPOCHS}e'

# Log training data for tensorboard
root_logdir = os.path.join(os.curdir, f"log/{name}/")
run_logdir = utils.get_run_logdir(root_logdir)
tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)

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

model = utils.create_densenet_model(NUM_CLASSES)
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
utils.visualize(history, txt, name)