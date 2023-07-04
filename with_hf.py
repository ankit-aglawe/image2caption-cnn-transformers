import tensorflow as tf
# import requests
# import numpy as np
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load pre-trained GPT-2 model and tokenizer
# model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc. depending on the desired model size
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Load pre-trained CNN model for feature extraction
# # Replace with the appropriate CNN model and weights
# cnn_model = tf.keras.applications.ResNet50(weights='imagenet')

# # Preprocess and extract image features
# def extract_image_features(image_path):
#     image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     image = tf.keras.preprocessing.image.img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = tf.keras.applications.resnet50.preprocess_input(image)
#     features = cnn_model.predict(image)
#     return features.flatten()

# # Generate captions using GPT-2 model
# def generate_captions(image_features, prompt):
#     encoded_input = tokenizer.encode(prompt, return_tensors='pt')
#     captions = []
#     with tf.device('cpu'):  # Run on CPU
#         for _ in range(5):  # Generate 5 captions
#             output = model.generate(encoded_input, max_length=50, num_return_sequences=1)
#             caption = tokenizer.decode(output[0], skip_special_tokens=True)
#             captions.append(caption)
#     return captions

# # Example usage
# image_path = 'test.jpg'
# prompt = 'Describe the image:'
# image_features = extract_image_features(image_path)
# captions = generate_captions(image_features, prompt)

# # Print the generated captions
# for i, caption in enumerate(captions):
#     print(f'Caption {i+1}: {caption}')


# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input

# # Load pre-trained GPT-2 model and tokenizer
# model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', etc. depending on the desired model size
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)

# # Fine-tune the model on your image captioning dataset
# # Replace with your own fine-tuning code
# # ...

# # Load pre-trained CNN model for feature extraction
# # Replace with the appropriate CNN model and weights
# cnn_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

# # Preprocess and extract image features
# def extract_image_features(image_path):
#     img = image.load_img(image_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = preprocess_input(x)
#     x = tf.expand_dims(x, axis=0)
#     features = cnn_model.predict(x)
#     return features

# # Generate captions using the fine-tuned model and visual features
# def generate_captions(image_features, prompt):
#     encoded_prompt = tokenizer.encode(prompt, return_tensors='pt')
#     with torch.no_grad():
#         # Generate captions conditioned on the image features
#         outputs = model.generate(
#             input_ids=encoded_prompt,
#             attention_mask=torch.ones_like(encoded_prompt),
#             max_length=50,
#             num_return_sequences=5,
#             do_sample=True,
#             temperature=0.8
#         )
#     captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
#     return captions

# # Example usage
# image_path = 'test.jpg'
# prompt = 'Describe the image:'
# image_features = extract_image_features(image_path)
# captions = generate_captions(image_features, prompt)

# # Print the generated captions
# for i, caption in enumerate(captions):
#     print(f'Caption {i+1}: {caption}')
# import torch
# from transformers import ViTGPT2LMHeadModel, ViTGPT2Tokenizer

# # Load the model and tokenizer
# model_name = "nlpconnect/vit-gpt2-image-captioning"
# model = ViTGPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = ViTGPT2Tokenizer.from_pretrained(model_name)

# # Set the device to use (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load and preprocess the image
# image_path = "test.jpg"
# # Preprocess the image as required by the model (e.g., resizing, normalization)

# # Generate captions for the image
# inputs = tokenizer(image_path, return_tensors="pt").input_ids.to(device)
# outputs = model.generate(inputs)

# # Decode the generated captions
# captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# for caption in captions:
#     print(caption)



from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds


preds= predict_step(['test3.jpeg']) # ['a woman in a hospital bed with a woman in a hospital bed']

print(preds)