from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter import ttk
from tkinter import filedialog  # Added filedialog for saving images
from PIL import Image, ImageDraw        # pip install pillow
from datetime import datetime
import cv2
import typing
import numpy as np

from spellchecker import SpellChecker

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

import os
from mltu.tensorflow.dataProvider import DataProvider    
import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.annotations.images import CVImage



class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        
        # image = image.astype(np.float32) / 255.0

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202311290851/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

# Import your existing ML-related modules and classes here...

class ImageToWordGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwriting Recognition")
        self.root.geometry("1100x420+150+50")
        self.root.configure(bg="#f2f3f5")
        self.root.resizable(False, False)

        self.current_x = 0
        self.current_y = 0
        self.color = 'black'
        self.image_counter = 0
        self.undo_list = []
        self.suggestions = []

        self.canvas = Canvas(root, width=930, height=300, background="white", cursor="hand2", bd=2, relief="solid")
        self.canvas.place(x=130, y=10)

        self.canvas.bind('<Button-1>', self.locate_xy)
        self.canvas.bind('<B1-Motion>', self.add_line)

        ttk.Style().configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="black")
        self.submit_button = ttk.Button(root, text='Submit', command=self.save_image)
        self.submit_button.place(x=30, y=100)

        self.eraser_button = ttk.Button(root, text='Erase All', command=self.new_canvas)
        self.eraser_button.place(x=30, y=180)

        self.undo_button = ttk.Button(root, text='Undo', command=self.undo_last)
        self.undo_button.place(x=30, y=260)

        self.prediction_label = Label(root, text="Prediction: ", bg="#f2f3f5", font=("Helvetica", 20))
        self.prediction_label.place(x=420, y=330)

        self.suggestions_label = Label(root, text=f"Suggested Corrections: {', '.join(self.suggestions)}", bg="#f2f3f5", font=("Helvetica", 12))
        self.suggestions_label.place(x=420, y=380)

        self.configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202311290851/configs.yaml")
        self.model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)
        

    def new_canvas(self):
        self.canvas.delete('all')

    def locate_xy(self, event):
        self.current_x = event.x
        self.current_y = event.y

    def add_line(self, event):
        line = self.canvas.create_line((self.current_x, self.current_y, event.x, event.y), width=8, fill=self.color, capstyle=ROUND, smooth=TRUE)
        self.current_x, self.current_y = event.x, event.y
        self.undo_list.append(line)

    def undo_last(self):
        if self.undo_list:
            last_drawn = self.undo_list[-20:]
            self.canvas.delete(*last_drawn)
            self.undo_list[:] = self.undo_list[:-20]

    def save_image(self):
        self.image_counter += 1

        img = Image.new('RGB', (self.canvas.winfo_width(), self.canvas.winfo_height()), 'white')
        draw = ImageDraw.Draw(img)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            draw.line(coords, fill='black', width=8)

        # Perform preprocessing on the drawn image
        processed_img = self.preprocess_image(img)

        # Pass the processed image through the model for prediction
        prediction_text = self.predict_text(processed_img)
        self.prediction_label.config(text=f"Prediction: {prediction_text}")

        print("Prediction: ", prediction_text)

        spell = SpellChecker()
        self.suggestions = list(spell.candidates(prediction_text))[:5]
        self.suggestions_label.config(text=f"Suggested Words: {', '.join(self.suggestions)}")



    def preprocess_image(self, img):
        # Add any necessary preprocessing steps here
        processed_img = img.resize((configs.width, configs.height))  # Adjust size as needed
        return processed_img

    def predict_text(self, img):
        # Use the existing ImageToWordModel class for prediction
        prediction_text = model.predict(np.array(img))
        return prediction_text


if __name__ == "__main__":
    root = Tk()
    app = ImageToWordGUI(root)
    root.mainloop()