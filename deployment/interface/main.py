import gradio as gr
import numpy as np
import cv2

from modeling.inference import MNISTInference


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def predict(img):
    img = cv2.resize(img, (28, 28))
    img = np.array(rgb2gray(img), dtype=np.float32)
    return str(model.predict(img))

model = MNISTInference('weights/mnist_model.pt')
iface = gr.Interface(predict, "image", "text")
iface.launch()