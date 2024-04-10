import requests
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow messages (1 = INFO, 2 = WARNING, 3 = ERROR)

from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Define function to generate captions for an image
def generate_captions(image_path):
    # Load the image
    raw_image = Image.open(image_path).convert("RGB")
    
    # Conditional image captioning (with additional text)
    additional_text = "a beautiful landscape with mountains"
    inputs = processor(raw_image, additional_text, return_tensors="pt")
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
    
    return conditional_caption, unconditional_caption

# Example usage
if __name__ == "__main__":
    img_path = r"C:\Envision\Implement\img.jpg"  # Update with the correct path to your image
    conditional_caption, unconditional_caption = generate_captions(img_path)
    print("Conditional Caption:", conditional_caption)
    print("Unconditional Caption:", unconditional_caption)
