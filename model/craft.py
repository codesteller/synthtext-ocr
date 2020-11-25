import tensorflow as tf 
import os


class CraftModel:
    def __init__(self):
        self.dbpath = "/home/codesteller/datasets/02_Business_Card/02_SynthText/SynthText/OCR_Data/ocr_gt"
        self.exp_dir = "../experiments/"
        self.exp_name = "craft_new_512"
        self.exp_path = os.path.join(self.exp_dir, self.exp_name)

    def model(self):
        pass