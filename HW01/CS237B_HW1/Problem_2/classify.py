import argparse, pdb
import numpy as np, tensorflow as tf
from utils import IMG_SIZE, LABELS, image_generator


def classify(model, test_dir):
    """
    Classifies all images in test_dir
    :param model: Model to be evaluated
    :param test_dir: Directory including the images
    :return: None
    """
    test_img_gen = image_generator.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        classes=LABELS,
        batch_size=1,
        shuffle=False,
    )

    ######### Your code starts here #########
    # Classify all images in the given folder
    # Calculate the accuracy and the number of test samples in the folder
    # test_img_gen has a list attribute filenames where you can access the
    # filename of the datapoint
        # break after 1 epoch
                #pdb.set_trace()
    num_test = test_img_gen.samples
    missclassified = []
    for ii in range(num_test):
        image, label = next(test_img_gen)
        true_index = np.argmax(label)
        scores = model.predict(image)
        predict_index = np.argmax(scores)
        if predict_index != true_index:
            if  test_img_gen.filenames[ii] not in missclassified:
                missclassified.append(test_img_gen.filenames[ii])

    print(missclassified)
    accuracy = model.evaluate(test_img_gen)[1]

    ######### Your code ends here #########

    print(f"Evaluated on {num_test} samples.")
    print(f"Accuracy: {accuracy*100:.0f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, default="datasets/test/")
    FLAGS, _ = parser.parse_known_args()
    model = tf.keras.models.load_model("./trained_models/trained.h5")
    classify(model, FLAGS.test_image_dir)
