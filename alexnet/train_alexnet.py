import tensorflow as tf
import time
import os
from tensorboard import program
import utils

# Define the constants
BATCH_SIZE = 16
EPOCHS = 5

(train_ds, test_ds, valid_ds, NUM_CLASSES) = utils.get_dataset('flowers', BATCH_SIZE)

i=0
tic = time.perf_counter()
name = f'alexnet_{i}_{NUM_CLASSES}c_{EPOCHS}e'

# Log training data for tensorboard
logdir = os.path.join("logs", time.strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir])
url = tb.launch()
print(f"Tensorflow listening on {url}")
# tensorboard --logdir=logs/ --host localhost --port 6006

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

model = utils.create_alexnet_model(NUM_CLASSES)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
history = model.fit(train_ds,
      epochs=EPOCHS,
      validation_data=valid_ds,
      validation_freq=1,
      callbacks=[tensorboard_callback, cp_callback])
model.save(name + '.h5')
toc = time.perf_counter()
print(f"Trained the dataset in {toc - tic:0.1f} seconds")
txt = f"{name} ({toc - tic:0.1f} s)"
utils.visualize(history, txt, name)