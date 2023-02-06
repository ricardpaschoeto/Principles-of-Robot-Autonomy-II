#!/usr/bin/env python

import argparse
import tensorflow as tf 
import pdb 
import os 

from model import build_model, build_baseline_model, loss
import utils

SIZE_BATCH = 32
LEARNING_RATE=0.0001
NUM_EPOCHS = 50

PATH_CHECKPOINT = os.path.join('trained_models', 'cp-{epoch:03d}.ckpt')
DIR_MODEL = 'trained_models'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    args = parser.parse_args()

    # Load dataset
    ramp_dataset_path = os.path.join('phys101', 'scenarios', 'ramp')
    train_dataset, test_dataset = utils.load_dataset(ramp_dataset_path,
                                                     ramp_surface=1,  # Choose ramp surface in experiments (1 or 2)
                                                     size_batch=SIZE_BATCH)

    # Build model
    if args.baseline:
        model = build_baseline_model()
        path_model = os.path.join(DIR_MODEL, "trained_baseline.h5")
    else:
        model = build_model()
        path_model = os.path.join(DIR_MODEL, "trained.h5")

    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                  loss=loss)

    # Checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=PATH_CHECKPOINT,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=5)

    # Tensorboard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='train_logs', update_freq='batch')

    model.summary()

    # Train model
    model.fit(train_dataset,
              epochs=NUM_EPOCHS,
              validation_data=test_dataset,
              steps_per_epoch=20,
              callbacks=[tensorboard_callback, cp_callback])

    model.save(path_model)
