import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import time


class EmbedModel:
    def __init__(self):
        checkpoint = "sam_vit_h_4b8939.pth"  # weights
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device="cpu")
        self.predictor = SamPredictor(sam)

    def embed(self, src: str):
        print("Embedding...", flush=True)
        image = cv2.imread("opm.jpeg")
        self.predictor.set_image(image)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        return image_embedding


start_time = time.time()
model = EmbedModel()
print(model.embed("opm.jpeg"))
end_time = time.time()
print("Time taken: ", end_time - start_time)
