from io import BytesIO
from core.logger import logger
from config import load_models
import numpy as np
from PIL import Image


class Digits:

    def __init__(self):
        self.model = load_models()
        logger.info("Digits model loaded")

    def preprocessing_digits(self, image):
        logger.info("Starting image preprocessing")
        img = Image.open(BytesIO(image)).convert("L")
        logger.info(f"Original image size: {img.size}")
        img = 255 - np.array(img)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((8, 8))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0 * 16
        img = img.reshape(1, 64)
        logger.info("Image preprocessing completed")
        return img

    def show_digits(self, image):
        logger.info("Showing digits image processed")
        img = Image.open(BytesIO(image)).convert("L")
        img = 255 - np.array(img)
        img = Image.fromarray(img.astype(np.uint8))
        img = img.resize((8, 8))
        return np.array(img, dtype=np.float32)

    def predict(self, image):
        logger.info("Prediction started")
        x = self.preprocessing_digits(image)
        prediction = self.model.predict(x)
        logger.info(f"Prediction result: {prediction[0]}")
        return prediction[0]