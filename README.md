# Excercise 1 - Hero Name Recognition

# Project Structure

```
Excercise_1
      |
      |---core
      |     |--best_model_state.pth
      |     |--detect.py
      |     |--getlabel.py
      |     |--img2vec.py
      |     |--main.py
      |     |--output.txt
      |
      |---cropped_images        # contain cropped and filter image (image after detect and before go to recognition model)
      |
      |---heros_images          # contain all heros image collect from official page of Wild Rift
      |
      |---test_data
      |     |--test_images
      |     |--heros_name.txt
      |     |--test.txt
      |
      |---train_backbone
      |     |--datasets
      |           |--images
      |           |--labels.csv
      |     |--acc.png
      |     |--loss.png
      |     |--best_model_state.pth
      |     |--data.py
      |     |--train.py
      |
      |--.gitignore
      |---README.md
      |---requirements.txt
      |---Take-Home-Exercise.pdf
```

# Environment requirements
The project run on Ubuntu 20.04 environment with python 3.8.10. To create virtual python environment for this project, follow this:
```
python3 -m venv env
source env/bin/activate
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

# Usages
## Reconstruction training backbone process

Step 1: go to the `train_backbone` folder `cd train_backbone`

Step 2: go on terminal and type `python3 train.py`

## Detect Heros

Step 1: go to `core` folder

Step 2: modity `main.py` with 2 params:
- `LABEL_DIR` : label of test images with format like `test_data/test.txt, if NOT, set `LABEL_DIR = ""`\
- `TEST_IMAGE_DIR` : folder contain all test images. \

By default:
```
LABEL_DIR = "../test_data/test.txt"
TEST_IMAGE_DIR = "../test_data/test_images"
```

Step 3: run `python3 main.py`. The program will return `output.txt` file with format:
```
image_name predict_hero similarity
```
