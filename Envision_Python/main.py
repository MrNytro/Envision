import os
from model_utils import load_vgg_model, preprocess_image
from text_utils import clean_text, generate_caption
from data_utils import load_captions, create_caption_mapping, tokenize_text

def main():
    image_path = 'img.jpg'
    captions_path = 'captions.txt'
    model_path = 'best_model.h5'

    vgg_model = load_vgg_model()

    captions_doc = load_captions(captions_path)
    mapping = create_caption_mapping(captions_doc)
    clean_text(mapping)
    tokenizer, vocab_size, max_length = tokenize_text(mapping)

    model = load_model(model_path)
    image = preprocess_image(image_path)
    caption = generate_caption(model, image, tokenizer, max_length)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()
