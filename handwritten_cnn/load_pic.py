import cv2.cv2 as cv
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import os
def load_pic(img_path, num):
    for step, pic_path in enumerate(glob.glob(img_path)):
        pic = cv.imread(pic_path)
        # pic = cv.cvtColor(pic,cv.COLOR_RGB2GRAY)
        pic = cv.resize(pic,(64, 64),interpolation=cv.INTER_CUBIC)
        pic = pic[np.newaxis, ...]

        if step == 0:
            data = pic
            label_num = [num]
        else:
            data = np.concatenate((data, pic), axis=0)
            label_num.append(num)
    return data, label_num

def load_num():
    for num in range(10):

        path_1 = os.getcwd()
        path_2 = r'\*.bmp'
        path = path_1 + '\handwritten' + '\\' + str(num) + path_2
        number, label = load_pic(path, num)
        label = np.array(label)
        if num == 0:
            data = number
            label_all = label
        else:
            data = np.concatenate((data, number), axis=0)
            label_all = np.hstack((label_all, label))
    return data, label_all
def load_data(size = 0.2):
    data, label = load_num()
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=size, random_state=42) # 数据按%80，%20划分
    return X_train, X_test, y_train, y_test

