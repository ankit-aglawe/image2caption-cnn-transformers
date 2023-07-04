

import tensorflow as tf
import requests
import numpy as np
import openai

# Set up OpenAI API credentials
openai.api_key = ''

# Load pre-trained CNN model for feature extraction
# Replace with the appropriate CNN model and weights
cnn_model = tf.keras.applications.ResNet50(weights='imagenet')

# Preprocess and extract image features
def extract_image_features(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    features = cnn_model.predict(image)
    return features.flatten()

# Generate captions using OpenAI's language model
def generate_captions(image_features, prompt):
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=50,
        temperature=0.8,
        n=5,  # Generate multiple caption variations
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        log_level="info"
    )
    captions = [choice['text'].strip() for choice in response.choices]
    return captions

# Example usage
image_path = 'test.jpg'
prompt = 'Describe the image:'
image_features = extract_image_features(image_path)
captions = generate_captions(image_features, prompt)

# Print the generated captions
for i, caption in enumerate(captions):
    print(f'Caption {i+1}: {caption}')
