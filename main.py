import os
from dotenv import load_dotenv
load_dotenv()

from mtgscan.text import MagicRecognition
from mtgscan.ocr.azure import Azure


if __name__ == "__main__":
    azure = Azure()
    rec = MagicRecognition(file_all_cards="all_cards.txt", file_keywords="Keywords.json")
    box_texts = azure.image_to_box_texts("https://assets-prd.ignimgs.com/2025/07/25/rarest-mtg-cards-1753454598010.jpg")
    deck = rec.box_texts_to_deck(box_texts)
    for card_name, count in deck:
        print(card_name, count)