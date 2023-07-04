from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

class FeatureExtractor:
    def __init__(self, model_name):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    def extract_features(self, images):
        inputs = self.feature_extractor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values
        return pixel_values
