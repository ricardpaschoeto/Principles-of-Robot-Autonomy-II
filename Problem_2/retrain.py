import argparse
import numpy as np
import tensorflow as tf1
import tensorflow.compat.v2 as tf
from utils import IMG_SIZE, image_generator, LABELS, maybe_makedirs
tf1.compat.v1.enable_eager_execution()

BATCH_SIZE = 100
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
lr = 0.01


def get_bottleneck_dataset(model, img_dir, img_size):
    # image_generator is of type ImageDataGenerator
    # Find more information here
    # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?version=stable
    train_img_gen = image_generator.flow_from_directory(img_dir,
                                                        target_size=(img_size, img_size),
                                                        shuffle=False,  # this is important for accessing filenames
                                                        classes=LABELS,
                                                        batch_size=1)

    bottleneck_x_l = []
    bottleneck_y_l = []
    for i in range(train_img_gen.samples):
        ######### Your code starts here #########
        # We want to create a dataset of bottleneck data.
        # You can get the next image input and one-hot encoded label via x_i, y_i = next(train_img_gen)
        # For each iteration append the bottleneck output as well as the label to the respective list
        # bottleneck_x_l -> list of tensors with dimension [1, bottleneck_size]
        # bottleneck_y_l -> list of tensors with dimension [1, num_labels]

        



        ######### Your code ends here #########

    bottleneck_ds = tf.data.Dataset.from_tensor_slices((np.vstack(bottleneck_x_l),
                                                        np.vstack(bottleneck_y_l)))

    return bottleneck_ds, train_img_gen.samples


def retrain(image_dir):
    # Create the base model from the pre-trained model InceptionV3
    base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   pooling='avg',
                                                   weights='imagenet')

    base_model.summary()
    base_model.compile(loss='mse')

    print("Generating Bottleneck Dataset... this may take some minutes.")
    bottleneck_train_ds, num_train = get_bottleneck_dataset(base_model, img_dir=image_dir, img_size=IMG_SIZE)
    train_batches = bottleneck_train_ds.shuffle(10000).batch(BATCH_SIZE).repeat()
    print("Done generating Bottleneck Dataset")

    ######### Your code starts here #########
    # We want to create a linear classifier which takes the bottleneck data as input
    # 1. Get the size of the bottleneck tensor. Hint: You can get the shape of a tensor via tensor.get_shape().as_list()
    # 2. Define a new tf.keras Model which is a linear classifier
    #   2.1 Define a keras Input (retrain_input)
    #   2.2 Define the trainable layer (retrain_layer) and name it "classifier" using the name argument
    #   2.3 Define the activation function (retrain activation)
    #   2.4 Create a new model
    # 3. Define a loss and a evaluation metric

    






    ######### Your code ends here #########

    retrain_model.compile(optimizer=tf.keras.optimizers.SGD(lr=lr),
                          loss=loss,
                          metrics=[metric])

    retrain_model.summary()

    EPOCHS = 1
    steps_per_epoch = 5000
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='retrain_logs', update_freq='batch')
    retrain_model.fit(train_batches,
                      epochs=EPOCHS,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=[tensorboard_callback])

    ######### Your code starts here #########
    # We now want to create the full model using the newly trained classifier
    # Use tensorflow keras Sequential to stack the base_model and the new layers
    



    
    ######### Your code ends here #########

    model.compile(loss=loss, metrics=[metric])

    maybe_makedirs('trained_models')
    model.save('trained_models/trained.h5')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    FLAGS, _ = parser.parse_known_args()
    retrain(FLAGS.image_dir)
