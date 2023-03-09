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

HERO_DIR = "../heros_images"

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
    
image_path = "../test_data/test_images/Annie_eYYaXcjRjQo_round36_Miss-Fortune_05-20-2021.mp4_39_1.jpg"
image = detect(image_path)
image = cv2.resize(image, (60, 60), interpolation=cv2.INTER_CUBIC)

# create a mask of the circle
circle_filter = np.zeros_like(image)
cv2.circle(circle_filter, (30, 30), 30, (255, 255, 255), -1)
image = cv2.bitwise_and(image, circle_filter)

image = Image.fromarray(image)
plt.imshow(image)
plt.show()
image_embeded = img2vec.get_vec(image)

similarity_list = {}
for hero, embeded in heros_embeded.items():
    similarity = cosine_similarity(image_embeded.reshape((1, -1)), embeded.reshape((1, -1)))
    print(f"Hero: {hero} - Similarity: {similarity}")
    similarity_list[hero] = similarity
    
# find the 5 heroes with the highest similarity
top5_heros = sorted(similarity_list, key=similarity_list.get, reverse=True)[:5]
print(top5_heros)