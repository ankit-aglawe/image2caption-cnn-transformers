from generator import GenerateCaptions
from preprocessing import ImageLoader
from feature_extractor import FeatureExtractor
import torch


model_name = "nlpconnect/vit-gpt2-image-captioning"

# Set the maximum caption length and number of beams for caption generation
max_length = 16
num_beams = 4


def get_captions(image_paths):
    images = ImageLoader.load_images(image_paths)
    feature_extractor = FeatureExtractor(model_name)
    generate_captions = GenerateCaptions(model_name, max_length, num_beams)

    pixel_values = feature_extractor.extract_features(images)

    captions = generate_captions.generate_captions(pixel_values)

    return captions



# # Define the model and tokenizer
# model_name = "nlpconnect/vit-gpt2-image-captioning"

# # Set the maximum caption length and number of beams for caption generation
# max_length = 16
# num_beams = 4

# # Test the image captioning on a single image
# image_paths = ['test.jpg']
# captions = get_captions(image_paths)
# print(captions)