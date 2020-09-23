from train import get_resnet_classification_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", metavar="<image_path>", help="image path"
    )
    args = parser.parse_args()
    return args.command


def predict(img_path, proc_data_path):
    num_classes = 3
    model, base_model = get_resnet_classification_model(num_classes)
    weights = proc_data_path + "models/" + 'finetune.h5'
    model.load_weights(weights, by_name=True)

    image = load_img(img_path, target_size=(224, 224))
    input_arr = img_to_array(image)
    input_arr /= 255
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    idx = np.argmax(predictions[0])
    label = ['not-a-checkbox', 'open-checkbox', 'checked-checkbox'][idx]
    print(label)
    return label


if __name__ == '__main__':
    img_path = parse()
    num_classes = 3
    proc_data_path = '../'
    predict(img_path, proc_data_path)
