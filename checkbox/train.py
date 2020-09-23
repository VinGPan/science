import keras
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import numpy as np
import argparse
from numpy.random import seed
from tensorflow import set_random_seed


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", metavar="<command>", help="'train_head' or 'finetune'"
    )
    args = parser.parse_args()
    return args.command


def get_call_backs(train_mode, proc_data_path):
    results_folder = proc_data_path + '/models/'
    # save_model = ModelCheckpoint(
    #     os.path.join(results_folder, train_mode + ".h5"),
    #     verbose=1, save_best_only=True, monitor="val_loss", mode="min")
    save_model = ModelCheckpoint(
        os.path.join(results_folder, train_mode + ".h5"),
        verbose=1, save_best_only=True, monitor="val_acc", mode="max")
    reduce_lr = ReduceLROnPlateau(
        monitor="loss", factor=0.1, patience=2, verbose=1, mode="auto", min_delta=0.0001, cooldown=0, min_lr=0)
    tb = TensorBoard(log_dir='../logs')
    return [save_model, reduce_lr, tb]


def train_resnet_classification(num_classes, train_mode, proc_data_path):
    seed(42)
    set_random_seed(42)
    model, base_model = get_resnet_classification_model(num_classes)
    lr = 0.001
    epochs = 30
    batch_size = 8
    steps = 100
    results_folder = proc_data_path + '/models/'
    if train_mode == 'train_head':
        for layer in base_model.layers:
            layer.trainable = False
    else:
        if train_mode == 'finetune':
            weights = results_folder + 'train_head' + ".h5"
            model.load_weights(weights, by_name=True)
        else:
            assert False, "unsupported train_mode"
        lr /= 100.0
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=lr, clipnorm=0.001),
        metrics=["acc"],
    )
    # model.summary()
    callbacks = get_call_backs(train_mode, proc_data_path)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        proc_data_path + '/data/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        proc_data_path + '/data/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
    # weights = np.array([1.0, 2.0, 3.0]) / 6.0
    model.fit_generator(
        train_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=20,
        callbacks=callbacks,
        # class_weight=weights
    )


def get_resnet_classification_model(num_classes):
    keras.backend.clear_session()
    base_model = ResNet50(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


if __name__ == '__main__':
    proc_data_path = '../'
    tmode = parse()
    num_classes = 3
    train_resnet_classification(num_classes, tmode, proc_data_path)
