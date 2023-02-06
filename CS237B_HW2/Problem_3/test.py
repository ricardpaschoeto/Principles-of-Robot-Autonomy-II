#!/usr/bin/env python

import argparse, os, pickle, re, pdb, glob

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf 

from model import build_model, build_baseline_model, loss, AccelerationLaw
import utils

SIZE_BATCH = 32

DIR_CHECKPOINT = 'trained_models' + os.sep
DIR_DATASET = os.path.join('phys101', 'scenarios', 'ramp')

def load_video(path_video):
    # Load video
    video = cv.VideoCapture(path_video)
    fps = video.get(cv.CAP_PROP_FPS)
    num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame[:1080//2,:1920//2])

    return frames

def show_image(frames, idx_frame, path_video, keypoints, params):
    color = (0,0,255)
    #search_path = os.path.join('phys101', 'scenarios', 'ramp', '(.*)', 'Camera_1.mp4')
    search_path = r'phys101/scenarios/ramp/(.*)/Camera_1.mp4'
    print(path_video)
    experiment = re.search(search_path, path_video).group(1)
    img = frames[idx_frame].copy()

    image_text = str(experiment) + ": " + str(idx_frame+1) + os.sep + str(len(frames))
    img = cv.putText(img, image_text, (50,70), cv.FONT_HERSHEY_DUPLEX, 1, color, 2)

    if 'mu_pred' in params:
        img = cv.putText(img, "a_pred: {:.3f}, a_groundtruth: {:.3f}, mu_pred: {:.3f}".format(params['a_pred'], params['a_groundtruth'], params['mu_pred']),
                         (50,100), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    else:
        img = cv.putText(img, "a_pred: {:.3f}, a_groundtruth: {:.3f}".format(params['a_pred'], params['a_groundtruth']),
                         (50,100), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    img = cv.putText(img, "mu_class", (750, 30), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    img = cv.putText(img, "p_class", (850, 30), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    if 'p_class' in params:
        for i, (mu, p) in enumerate(zip(params['mu_class'].tolist(), params['p_class'].tolist())):
            img = cv.putText(img, "{:.3f}".format(mu),
                             (750,55 + 15*i), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
            img = cv.putText(img, "{:.3f}".format(p),
                             (850,55 + 15*i), cv.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    img = cv.putText(img, "prev/next video: w/s, prev/next frame: a/d, quit: q", (50, 500), cv.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
    colors = [(0,255,0), (255,0,255)]

    m = re.search(r'([12]0)_0[12]', path_video)
    rad_slope = float(m.group(1)) * np.pi / 180.
    slope = np.array([np.cos(rad_slope), np.sin(rad_slope)])
    x = 0.5 * params['a_pred'] * idx_frame*idx_frame
    point_pred = keypoints[0][1] + x * slope
    img = cv.circle(img, (int(point_pred[0]), int(point_pred[1])), 10, (0,0,255), -1)
    for i, (_, point) in enumerate(keypoints):
        img = cv.circle(img, point, 10, colors[i], -1)
    cv.imshow('image', img)
    return cv.waitKey(0)

def handle_video(frames, path_video, keypoints, params):

    idx_frame = 0
    idx_keypoint = 0
    while True:
        idx_frame = max(0, min(len(frames) - 1, idx_frame))

        key = show_image(frames, idx_frame, path_video, keypoints, params)
        if key == ord('a'):  # left
            idx_frame -= 1
        elif key == ord('d'):  # right
            idx_frame += 1
        elif key == ord('x'):
            keypoints.clear()
        elif key in (ord('w'), ord('s'), ord('q')):  # up, down, q
            break
    return key

def handle_dataset(video_paths, keypoints, params):
    # Restore last video
    idx_video = 0

    while True:
        idx_video = max(0, min(len(video_paths) - 1, idx_video))

        path_video = video_paths[idx_video]
        frames = load_video(path_video)
        kp = keypoints[path_video]
        p = {
            'a_pred': params['a_pred'][idx_video][0],
            'a_groundtruth': params['a_groundtruth'][idx_video]
        }
        if 'mu_class' in params:
            p['mu_pred'] = params['mu_pred'][idx_video][0]
            p['mu_class'] = params['mu_class']
            p['p_class'] = params['p_class'][idx_video]

        key = handle_video(frames, path_video, kp, p)

        if key == ord('w'):  # up
            idx_video -= 1
        elif key == ord('s'):  # down
            idx_video += 1
        elif key == ord('q'):
            break
    return keypoints

def compute_accelerations(video_paths, keypoints):
    accelerations = []
    for kp, path_video in zip(keypoints, video_paths):
        if len(kp) != 2:
            accelerations.append(0.)
            continue

        # Compute slope from file path
        m = re.search(r'([12]0)_0[12]', path_video)
        rad_slope = float(m.group(1)) * np.pi / 180.
        slope = np.array([np.cos(rad_slope), np.sin(rad_slope)])

        # Compute acceleration
        # x_1 = 1/2 * a * t^2 + v_0 * t + x_0
        t_0, x_0 = kp[0]
        t_1, x_1 = kp[1]
        x_0 = np.array(x_0)
        x_1 = np.array(x_1)
        dt = t_1 - t_0
        a = 2. / (dt * dt) * (x_1 - x_0).dot(slope)

        accelerations.append(a)
    return accelerations

def load_keypoints(dir_dataset, ramp_surface):
    dataset_path = os.path.join('phys101', 'scenarios', 'ramp', '*', '*', '*', 'Camera_1.mp4')
    dataset_fnames = [fname.replace(os.sep, "/") for fname in glob.glob(dataset_path)]
    #dataset = tf.data.Dataset.list_files(dataset_path, shuffle=False)
    dataset = tf.data.Dataset.from_tensor_slices(dataset_fnames)
    size_dataset = tf.data.experimental.cardinality(dataset).numpy() - 1  # 1804 (w/o last video - broken)
    dataset = dataset.take(size_dataset)
    video_paths = [d.numpy().decode('utf-8') for d in dataset]

    if os.path.exists('keypoints.pkl'):
        with open('keypoints.pkl', 'rb') as f:
            keypoints = pickle.load(f)

    keypoints_dict = {}
    for path, kp in zip(video_paths, keypoints):
        keypoints_dict[path] = kp
    return keypoints_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', dest='baseline', action='store_true')
    args = parser.parse_args()

    # Load dataset
    ramp_surface = 1  # Choose ramp surface in experiments (1 or 2)
    _, test_dataset = utils.load_dataset(DIR_DATASET,
                                         ramp_surface=ramp_surface,
                                         size_batch=SIZE_BATCH,
                                         return_filenames=True)
    video_paths = [p.decode('utf-8') for p in np.concatenate([d[0][2] for d in test_dataset]).tolist()]
    a_groundtruth = np.concatenate([d[1] for d in test_dataset])

    # Load dataset again without the filenames 
    _, test_dataset = utils.load_dataset(DIR_DATASET,
                                         ramp_surface=ramp_surface,
                                         size_batch=SIZE_BATCH)
    # Build model
    if args.baseline:
        model = tf.keras.models.load_model(DIR_CHECKPOINT + 'trained_baseline.h5', custom_objects={'loss': loss})
        a_pred = model.predict(test_dataset)
        parameters = {
            'a_pred': np.maximum(0., a_pred),
            'a_groundtruth': a_groundtruth
        }
    else:
        model = tf.keras.models.load_model(DIR_CHECKPOINT + 'trained.h5', custom_objects={'AccelerationLaw': AccelerationLaw, 'loss': loss})
        # model.load_weights(tf.train.latest_checkpoint(DIR_CHECKPOINT))

        outputs = [model.output,
                   model.get_layer('p_class').output,
                   model.get_layer('mu').output,
                  ]
        model = tf.keras.Model(inputs=model.input, outputs=outputs)
        mu_class = model.get_layer('mu').get_weights()[0][:,0]
        g = model.get_layer('a').get_weights()[0]
        a_pred, p_class, mu_pred = model.predict(test_dataset)
        parameters = {
            'a_pred': np.maximum(0., a_pred),
            'a_groundtruth': a_groundtruth,
            'mu_pred': mu_pred,
            'p_class': p_class,
            'mu_class': mu_class
        }

        with open('debug.log', 'w') as f:
            f.write('p_class:\n')
            for i in range(p_class.shape[0]):
                f.write('{}\n'.format(p_class[i]))
            f.write('\nmu_class:\n{}\n'.format(mu_class.T))
            f.write('\nmu:\n{}\n'.format(mu_pred.T))
            f.write('\na:\n{}\n'.format(a_pred.T))
            f.write('\ng:\n{}\n'.format(g.T))

    keypoints = load_keypoints(DIR_DATASET, ramp_surface=ramp_surface)

    handle_dataset(video_paths, keypoints, parameters)

    cv.destroyAllWindows()
