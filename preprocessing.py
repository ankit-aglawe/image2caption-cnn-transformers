import cv2
import numpy as np

class ImageLoader:
    @staticmethod
    def load_images(image_paths, target_size=(224, 224)):
        images = []
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image = ImageLoader.resize_image(image, target_size)
            #image = ImageLoader.normalize_image(image)
            images.append(image)
        return images

    @staticmethod
    def resize_image(image, target_size):
        resized_image = cv2.resize(image, target_size)
        return resized_image

    @staticmethod
    def normalize_image(image):
        normalized_image = image / 255.0  # Normalize pixel values to [0, 1]
        return normalized_image

    @staticmethod
    def augment_image(image):
        # Apply data augmentation techniques here (e.g., random cropping, flipping, rotation)
        augmented_image = image  # Placeholder, replace with actual data augmentation
        return augmented_image
