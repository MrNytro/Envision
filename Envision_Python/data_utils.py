import os
import pickle
from tqdm.notebook import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer

def load_captions(file_path):
    with open(file_path, 'r') as f:
        next(f)
        captions_doc = f.read()
    return captions_doc

def create_caption_mapping(captions_doc):
    mapping = {}
    for line in tqdm(captions_doc.split('\n')):
        tokens = line.split(',')
        if len(line) < 2:
            continue
        image_id, caption = tokens[0], tokens[1:]
        image_id = image_id.split('.')[0]
        caption = " ".join(caption)
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(caption)
    return mapping

def tokenize_text(mapping):
    all_captions = []
    for key in mapping:
        for caption in mapping[key]:
            all_captions.append(caption)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer, len(tokenizer.word_index) + 1, max(len(caption.split()) for caption in all_captions)