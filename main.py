import os
from dotenv import load_dotenv
import card_capture
load_dotenv()

from mtgscan.text import MagicRecognition
from mtgscan.ocr.azure import Azure


if __name__ == "__main__":
    card_capture.run()