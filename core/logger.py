import os
import logging

os.makedirs("logs", exist_ok=True)
logger=logging.getLogger('Digits App')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/app.log')
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)

# Handler add karo
logger.addHandler(file_handler)