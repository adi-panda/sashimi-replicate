# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel
from model import EmbedModel
import numpy as np


class Output(BaseModel):
    data: list[float]
    shape: list[int]

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = EmbedModel()

    def predict(
        self,
        image: Path = Input(description="Grayscale input image"),
    ) -> Output:
        """Run a single prediction on the model"""
        result = self.model.embed(image)
        result_shape = result.shape
        result = result.flatten()
        return Output(data=result.tolist(), shape=list(result_shape))
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
