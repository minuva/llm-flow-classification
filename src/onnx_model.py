import numpy as np
import os
import json

from typing import List
from onnxruntime import InferenceSession


class OnnxTransformer:
    def __init__(self, model_dir: str, providers: List = ["CPUExecutionProvider"]):

        self.model_dir = model_dir
        self.config = self.load_config()
        self.onnx_model_name = self.find_onnx_model()
        self.model = InferenceSession(self.onnx_model_name, providers=providers)
        self.output_names = self.model.get_outputs()
        self.providers = providers
        self.output_names = [output.name for output in self.model.get_outputs()]
        self.input_names = [input.name for input in self.model.get_inputs()]
        self.tokenizer = None

    def load_config(self):
        with open(os.path.join(self.model_dir, "config.json"), "r") as f:
            config = json.load(f)

        return config

    def find_onnx_model(self):
        onnx_files = [
            file for file in os.listdir(self.model_dir) if file.endswith(".onnx")
        ]

        if not onnx_files:
            raise FileNotFoundError("No ONNX files found in the specified directory.")
        elif len(onnx_files) > 1:
            raise ValueError(
                "More than one ONNX file found in the specified directory."
            )

        return os.path.join(self.model_dir, onnx_files[0])

    def predict(self, tokenizer, texts, batch_size) -> List:
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for subtexts in np.array_split(np.array(texts), len(texts) // batch_size + 1):
            encodings = tokenizer.encode_batch(list(subtexts))
            inputs = {
                "input_ids": np.vstack(
                    [encoding.ids for encoding in encodings], dtype=np.int64
                ),
                "attention_mask": np.vstack(
                    [encoding.attention_mask for encoding in encodings], dtype=np.int64
                ),
                "token_type_ids": np.vstack(
                    [encoding.type_ids for encoding in encodings], dtype=np.int64
                ),
            }

            for input_name in self.input_names:
                if input_name not in inputs:
                    raise ValueError(f"Input name {input_name} not found in inputs")

            inputs = {input_name: inputs[input_name] for input_name in self.input_names}
            output = np.squeeze(
                np.stack(
                    self.model.run(output_names=self.output_names, input_feed=inputs)
                ),
                axis=0,
            )
            results.extend(output)

        return results
