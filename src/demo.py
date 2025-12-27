import tensorflow as tf
from HwTR import *
import numpy as np
import os

# Configure MODEL PATH
MODEL_PATH = "HwTR_BiLSTM.h5"
IMG_H = 64
IMG_W = 128
LETTERS = (
    [' '] +
    [str(d) for d in range(10)] +
    [chr(c) for c in range(ord('A'), ord('Z')+1)] +
    [chr(c) for c in range(ord('a'), ord('z')+1)]
)

hwr = HwTR(
    img_w=128,
    img_h=64,
    max_text_length=16,
    num_classes=len(LETTERS)+1,
    letters=LETTERS
)

if __name__ == "__main__":
    model, error = hwr.load(MODEL_PATH)
    if not error:
        test_image = "demo.png"
        if not os.path.exists(test_image):
            print(f"Image not found: {test_image}")
            exit()
        print(f"Result : {hwr.preprocess_and_recognize(test_image)}")
    else:
        print(f"Error load model")
        exit()