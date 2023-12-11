import cv2
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
# import time


class EmbedModel():
    def __init__(self):
        checkpoint = "sam_vit_h_4b8939.pth" # weights
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

    def embed(self, src):
        print("Embedding...", flush=True)
        print(src, flush=True)
        image = cv2.imread(str(src))
        self.predictor.set_image(image)
        image_embedding = self.predictor.get_image_embedding().cpu().numpy()
        return image_embedding
