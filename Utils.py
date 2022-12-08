import cv2 as cv
import numpy as np
import os

def resize_img(filepath):
    img_rgb = cv.imread(filepath)
    img_grey = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
    img_adapted = cv.adaptiveThreshold(img_grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 101, 9)
    img_stacked = np.repeat(img_adapted[...,None], 3, axis = 2)
    resized = cv.resize(img_stacked, (224, 224), interpolation=cv.INTER_AREA)
    bg_img = 255* np.ones(shape = (224,224, 3))
    bg_img[0:224, 0:224, :] = resized
    bg_img /= 255
    bg_img = np.rollaxis(bg_img, 2, 0)
    return bg_img

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def word2idx(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def load_val_images(data_dir):
    image_filenames = []
    images = []
    all_filenames = os.listdir(data_dir)
    all_filenames.sort()
    for filename in all_filenames:
        if filename[-3:] == 'png':
            image_filenames.append(filename)
    for name in image_filenames:
        image = resize_img(data_dir + name)
        images.append(image)
    return images