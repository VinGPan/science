import glob
from train import get_resnet_classification_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, return_cm=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2)

    if not title:
        title = "Confusion matrix"

    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print(cm)

    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    if return_cm:
        return cm

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.show()
    return ax


def report(proc_data_path):
    dpath = proc_data_path + '/data/test/'
    imgs_path = glob.glob(dpath+"*.png")

    num_classes = 3
    model, base_model = get_resnet_classification_model(num_classes)
    weights = proc_data_path + "/models/" + 'finetune.h5'
    model.load_weights(weights, by_name=True)
    labels = []
    preds = []
    classes = ['not-a-checkbox', 'open-checkbox', 'checked-checkbox']
    for img_path in imgs_path:
        # print("precessing image " + str(img_path))
        if img_path.find("not") != -1:
            label = 'not-a-checkbox'
        elif img_path.find("open") != -1:
            label = 'open-checkbox'
        else:
            label = 'checked-checkbox'
        image = load_img(img_path, target_size=(224, 224))
        input_arr = img_to_array(image)
        input_arr /= 255
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        predictions = model.predict(input_arr)
        idx = np.argmax(predictions[0])
        pred = classes[idx]
        labels.append(label)
        preds.append(pred)

    accuracy = round(accuracy_score(labels, preds) * 100, 1)
    res_str = ('Test accuracy = ' + str(accuracy) + '%')
    print(res_str)
    plot_confusion_matrix(labels, preds, classes)


if __name__ == '__main__':
    proc_data_path = '../'
    report(proc_data_path)
