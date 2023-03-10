import os
import glob

def get_label(label_path="../test_data/test.txt"):
    label_dict = {}
    with open(label_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.split("\t")
            image_name = line[0]
            label = line[1]
            label_dict[image_name] = label
    return label_dict