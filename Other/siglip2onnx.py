from PIL import Image
import requests
import os
import pandas as pd
from transformers import AutoProcessor, AutoModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    CLIPImageProcessor,
    CLIPVisionModel,
)
import torch
import numpy as np
import onnx
from typing import Any, Optional, Tuple, Union

arsenal_model_root = "/data/.modelcache/common-crawl-data/model-repo/"
visual_encoder_name_or_path = f"{arsenal_model_root}google/siglip-so400m-patch14-384/7067f6db2baa594bab7c6d965fe488c7ac62f1c8"


class ModifiedSigLIPModel(SiglipVisionModel):
    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


visual_encoder = (
    ModifiedSigLIPModel.from_pretrained(visual_encoder_name_or_path).eval().cuda()
)

imgs_size = [1, 3, 384, 384]
print(visual_encoder)
imgs = torch.ones(tuple(imgs_size)).cuda()
onnx_path = "siglip_hidden.onnx"

dynamic_axes = {"input": {0: "batch"}}
output_names = []

for i in range(30):
    output_names.append("hs_{}".format(i))
    dynamic_axes["hs_{}".format(i)] = {0: "batch"}
print(output_names)
print(dynamic_axes)


with torch.no_grad():
    torch.onnx.export(
        visual_encoder,
        imgs,
        onnx_path,
        verbose=False,
        opset_version=15,
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
