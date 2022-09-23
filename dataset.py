import numpy as np
from matplotlib import pylab as plt
import cv2
from PIL import Image
import os
import shutil

def add_noise(img, noise_quantity):
    img_copy = img.copy()
    img_values = np.array(img_copy)
    w, h, c = img_values.shape
    for i in range(noise_quantity):
        index1 = np.random.randint(0, w)
        index2 = np.random.randint(0, h)
        img_copy.putpixel((index2, index1), (0, 0, 0, 255 - img_values[index1][index2][3]))

    return img_copy

def create_folders():
    for letter in range(97, 123):
        parent_dir = os.path.join('/Users/mihailplesa/Documents/Doctorat/Research/Dataset/', str(chr(letter)))
        path = os.path.join(parent_dir, 'train')
        os.mkdir(path)
        path = os.path.join(parent_dir, 'test')
        os.mkdir(path)

def create_train_images(num_of_images):
    for letter in range(97, 123):
        parent_dir = os.path.join('/Users/mihailplesa/Documents/Doctorat/Research/Dataset/', str(chr(letter)))
        path = os.path.join(parent_dir, 'train')
        os.mkdir(path)
        img = Image.open(parent_dir + "/" + str(chr(letter)) + '.png')
        for noise_quantity in range(8):
            for idx in range(num_of_images):
                noisy_img = add_noise(img, noise_quantity)
                noisy_img_name = str(chr(letter)) + "_" + str(noise_quantity) + "_" + str(idx) + ".png"
                noisy_img_path = str(path) + "/" + noisy_img_name
                print(noisy_img_path)
                noisy_img.save(noisy_img_path)

def create_test_images(num_of_images):
    for letter in range(97, 123):
        parent_dir = os.path.join('/Users/mihailplesa/Documents/Doctorat/Research/Dataset/', str(chr(letter)))
        path = os.path.join(parent_dir, 'test')
        os.mkdir(path)
        img = Image.open(parent_dir + "/" + str(chr(letter)) + '.png')
        for noise_quantity in range(8):
            for idx in range(num_of_images):
                noisy_img = add_noise(img, noise_quantity)
                noisy_img_name = str(chr(letter)) + "_" + str(noise_quantity) + "_" + str(idx) + ".png"
                noisy_img_path = str(path) + "/" + noisy_img_name
                print(noisy_img_path)
                noisy_img.save(noisy_img_path)

def clear_folders():
    parent_dir = '/Users/mihailplesa/Documents/Doctorat/Research/Dataset/'
    for letter in range(97, 123):
        path = os.path.join(parent_dir, str(chr(letter)))
        train_path = os.path.join(path, 'train')
        test_path = os.path.join(path, 'test')
        if os.path.isdir(train_path) == True:
            shutil.rmtree(train_path)
        if os.path.isdir(test_path) == True:
            shutil.rmtree(test_path)


clear_folders()
create_train_images(20)
create_test_images(3)
