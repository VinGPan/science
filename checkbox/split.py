import glob
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from shutil import copyfile


def make_dir(dpath):
    try:
        os.makedirs(dpath)
    except:
        pass


def split(dpath, proc_data_path):
    make_dir(proc_data_path + 'models')
    make_dir(proc_data_path + 'data/test')
    make_dir(proc_data_path + 'data/train/0')  #not
    make_dir(proc_data_path + 'data/val/0')    #not
    make_dir(proc_data_path + 'data/train/1') #open
    make_dir(proc_data_path + 'data/val/1')   #open
    make_dir(proc_data_path + 'data/train/2') #checked
    make_dir(proc_data_path + 'data/val/2')   #checked

    imgs_path = glob.glob(dpath+"/**/*.png", recursive=True)
    labels = []
    for img in imgs_path:
        if img.find("not") != -1:
            labels.append(0)
        elif img.find("open") != -1:
            labels.append(1)
        else:
            labels.append(2)
    imgs_path = np.array(imgs_path)
    labels = np.array(labels)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sss.get_n_splits(imgs_path, labels)
    for train_index, test_index in sss.split(imgs_path, labels):
        X_train, X_test = imgs_path[train_index], imgs_path[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        t_cnt_per_cls = int((y_test.shape[0] / 3) / 2)

        tests = list(np.concatenate([(X_train[y_train == 0])[0:t_cnt_per_cls],
                                     (X_train[y_train == 1])[0:t_cnt_per_cls],
                                     (X_train[y_train == 2])[0:t_cnt_per_cls]]))

        for img_i, img in enumerate(X_train):
            img_name = (img.strip().replace("\\", "/").split("/"))[-1]
            if img_name[0] == '_':
                continue
            if img in tests:
                dst = proc_data_path + 'data/test/' + str(img_i) + "_" + img_name
            else:
                dst = proc_data_path + 'data/train/' + str(y_train[img_i]) + "/" + str(img_i) + "_" + img_name
            copyfile(img, dst)
        for img_i, img in enumerate(X_test):
            img_name = (img.strip().replace("\\", "/").split("/"))[-1]
            if img_name[0] == '_':
                continue
            dst = proc_data_path + 'data/val/' + str(y_test[img_i]) + "/" + str(img_i) + "_" + img_name
            copyfile(img, dst)


if __name__ == '__main__':
    dpath = '../checkbox-data/'
    proc_data_path = '../'
    split(dpath, proc_data_path)
