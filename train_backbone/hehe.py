import pandas as pd
import os
import glob
import shutil

labels = {
    "image_name": [],
    "label": []
}


SOURCE_DIR = "datasets/images"
OUT_PATH = "datasets/labels.csv"

image_list_name = os.listdir(SOURCE_DIR)
image_list_name.sort()

for i in range(len(image_list_name)):
    labels["image_name"].append(image_list_name[i])
    labels["label"].append(i)

labels = pd.DataFrame(labels)
labels.to_csv(OUT_PATH, index=False)
