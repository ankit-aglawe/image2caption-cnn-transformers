from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GenerateCaptions:
    def __init__(self, model_name, max_length, num_beams):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.num_beams = num_beams

    def generate_captions(self, pixel_values):
        output_ids = self.model.generate(pixel_values, max_length=self.max_length, num_beams=self.num_beams)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds