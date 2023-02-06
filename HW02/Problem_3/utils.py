import re, glob, os, pdb

import numpy as np
import tensorflow as tf 

def load_accelerations(filename, dataset):
    accelerations_dict = {}
    with open(filename) as f:
        for line in f:
            hardcoded_path_video, a = line.split('\t')
            path_video = hardcoded_path_video
            accelerations_dict[path_video] =float(a)

    accelerations = []
    for d in dataset:
        path_video = d.numpy().decode('utf-8')
        accelerations.append(accelerations_dict[path_video])
    return np.array(accelerations, dtype=np.float32)

def parse_angles(dataset):
    angles = []
    for d in dataset:
        path_video = d.numpy().decode('utf-8')
        m = re.search(r'([12]0)_0[12]', path_video)
        rad_slope = float(m.group(1)) * np.pi / 180.
        angles.append(rad_slope)
    return np.array(angles, dtype=np.float32)

def get_initial_video_frame(d):
    path_video = d
    path_img = tf.strings.substr(path_video, 0, tf.strings.length(path_video) - 4)
    path_img = tf.strings.regex_replace(path_img, '/', '-')
    path_img = tf.strings.join(['frames', os.sep, path_img, '.jpg'])
    img = tf.io.read_file(path_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def load_dataset(path_experiment, ramp_surface=1, size_batch=1, return_filenames=False):
    # List all video files
    path = os.path.join(path_experiment, "*", "*0_0{}".format(ramp_surface), "*", "Camera_1.mp4")
    fnames = glob.glob(path)
    fnames = [fname.replace(os.sep, "/") for fname in fnames]
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    # Compute size of dataset
    size_dataset = tf.data.experimental.cardinality(dataset).numpy() - 1  # 1804 (w/o last video - broken)
    num_batches = int((size_dataset + 0.5) / size_batch)
    num_test = num_batches // 5  # 1
    num_train = num_batches - num_test  # 14
    dataset = dataset.take(size_dataset)

    # Load acceleration labels
    accelerations = tf.data.Dataset.from_tensor_slices(load_accelerations('accelerations.log', dataset))
    list_accel = list(accelerations.as_numpy_iterator())
    parsed_angles = parse_angles(dataset)
    # angles = tf.data.Dataset.from_tensor_slices(parse_angles(dataset))
    angles = tf.data.Dataset.from_tensor_slices(parsed_angles)

    # Load first frame from each video and normalize
    images = dataset.map(get_initial_video_frame) \
                    .map(tf.image.per_image_standardization)
    if return_filenames:
        input_tuple = (images, angles, dataset)
    else:
        input_tuple = (images, angles)
    input_set = tf.data.Dataset.zip(input_tuple)
    output_set = accelerations
    dataset = tf.data.Dataset.zip((input_set, output_set)) \
                  .shuffle(size_dataset, reshuffle_each_iteration=False, seed=1234) \
                  .batch(size_batch) \

    # Create train and test datasets
    test_dataset = dataset.take(num_test)
    train_dataset = dataset.skip(num_test) \
                           .shuffle(num_train, reshuffle_each_iteration=True) \
                           .repeat(None)
    return train_dataset, test_dataset

