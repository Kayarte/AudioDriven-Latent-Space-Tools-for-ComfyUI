import torch
from typing import List, Dict, Tuple

class TimestampNoiseGenerator:
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "noise_params": ("NOISE_PARAMS",),
               "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
               "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
               "noise_type": (["gaussian", "salt_pepper", "perlin"],),
               "analysis_type": ("ANALYSIS_TYPE",),
           }
       }

   RETURN_TYPES = ("LATENT", "TIMESTAMPS")
   FUNCTION = "generate_timestamp_noise"
   CATEGORY = "audio/noise"

   def __init__(self):
       pass

   def generate_timestamp_noise(self, noise_params, width, height, noise_type, analysis_type):
       timestamps = noise_params.get("timestamps", [])
       if len(timestamps) == 0:
           return ({"samples": torch.zeros((1, 4, height//8, width//8))}, [0.0])
       
       batch_size = len(timestamps)  # Use all measurements
       latent_height, latent_width = height//8, width//8
       noise_batch = torch.zeros((batch_size, 4, latent_height, latent_width))

       base_intensity = noise_params[noise_type]["intensity"]
       base_persistence = noise_params[noise_type]["persistence"] 
       base_grain = noise_params[noise_type]["grain"]

       for i, timestamp in enumerate(timestamps):
           time_scale = timestamp / timestamps[-1]
           rand_factor = torch.rand(1).item() * 0.3
           
           modified_params = {
               noise_type: {
                   "intensity": base_intensity * (1.0 + time_scale + rand_factor),
                   "persistence": base_persistence * (1.2 - time_scale * 0.3),
                   "grain": base_grain * (1.0 + time_scale * 0.8 + rand_factor)
               }
           }
           
           frame = self.generate_basic_noise(modified_params, latent_height, latent_width, noise_type, analysis_type)
           noise_batch[i] = frame[0]

       return ({"samples": noise_batch}, timestamps)

   def generate_basic_noise(self, params, height, width, noise_type, analysis_type):
       noise = torch.zeros((1, 4, height, width))
       
       if noise_type == "gaussian":
           noise = torch.randn((1, 4, height, width))
       elif noise_type == "salt_pepper":
           mask = torch.rand((1, 4, height, width)) < params[noise_type]["grain"]
           noise[mask] = params[noise_type]["intensity"]
           noise[~mask] = -params[noise_type]["intensity"]
       
       return noise * params[noise_type]["persistence"]

NODE_CLASS_MAPPINGS = {
   "TimestampNoiseGenerator": TimestampNoiseGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
   "TimestampNoiseGenerator": "Timestamp Noise Generator"
}