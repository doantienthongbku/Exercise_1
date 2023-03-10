import os
import glob

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity, manhattan_distances, euclidean_distances
import numpy as np
import torch.nn as nn
from scipy.spatial import distance
import matplotlib.pyplot as plt
import cv2

from img2vec import Img2Vec
from detect import detect
from getlabel import get_label

HERO_DIR = "../heros_images"
LABEL_DIR = "../test_data/test.txt"
TEST_IMAGE_DIR = "../test_data/test_images"

if LABEL_DIR == "":
    label_data = None
else:
    label_data = get_label(LABEL_DIR)

img2vec = Img2Vec(cuda=False)

heros_list =  glob.glob(os.path.join(HERO_DIR, '*.png'))
heros_embeded = {}
for hero in heros_list:
    hero_name = os.path.basename(hero).replace('.png', '')
    
    hero_image = cv2.imread(hero)
    circle_filter = np.zeros_like(hero_image)
    cv2.circle(circle_filter, (30, 30), 30, (255, 255, 255), -1)
    hero_image = cv2.bitwise_and(hero_image, circle_filter)
    
    hero_image = Image.fromarray(hero_image)
    hero_embeded = img2vec.get_vec(hero_image)
    heros_embeded[hero_name] = hero_embeded

number_true_image_top1 = 0
number_true_image_top5 = 0
string_write_to_output = ""

image_name_list = os.listdir(TEST_IMAGE_DIR)
image_name_list.sort()
for image_name in image_name_list:
    image_path = os.path.join(TEST_IMAGE_DIR, image_name)
    image = detect(image_path)
    image = cv2.resize(image, (60, 60), interpolation=cv2.INTER_CUBIC)

    # create a mask of the circle
    circle_filter = np.zeros_like(image)
    cv2.circle(circle_filter, (30, 30), 30, (255, 255, 255), -1)
    image = cv2.bitwise_and(image, circle_filter)

    image = Image.fromarray(image)
    image_embeded = img2vec.get_vec(image)

    similarity_list = {}
    for hero, embeded in heros_embeded.items():
        similarity = cosine_similarity(image_embeded.reshape((1, -1)), embeded.reshape((1, -1)))
        similarity_list[hero] = similarity

    # get the top 5 hero with the highest similarity
    top_5_heros = sorted(similarity_list, key=similarity_list.get, reverse=True)[:5]
    
    # get the name of hero with the highest similarity
    hero_name_pred = top_5_heros[0]
    
    if label_data is not None:
        hero_name_true = label_data[image_name]
        print(image_name)
        print(f"Pred: {hero_name_pred} - {hero_name_true}")
    else:
        print(image_name)
        print(f"Pred: {hero_name_pred}")
        
    # print("Similarity: ", similarity_list[hero_name_pred][0][0])
        
    string_write_to_output += f"{image_name} {hero_name_pred} {similarity_list[hero_name_pred][0][0]:.4f}\n"
    
    if hero_name_pred == hero_name_true:
        number_true_image_top1 += 1
    if hero_name_true in top_5_heros:
        number_true_image_top5 += 1

open("output.txt", "w").write(string_write_to_output)
if label_data is not None:
    print(f"Accuracy top 1: {number_true_image_top1} - {len(image_name_list)} ({number_true_image_top1/len(image_name_list)*100}%))")
    print(f"Accuracy top 5: {number_true_image_top5} - {len(image_name_list)} ({number_true_image_top5/len(image_name_list)*100}%))")
