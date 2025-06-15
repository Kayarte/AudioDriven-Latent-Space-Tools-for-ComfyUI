import torch
import math
from .audio_noise_nodes import NoiseParams

class NoiseToLatentConverter:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_params": ("NOISE_PARAMS",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "noise_type": (["gaussian", "salt_pepper", "perlin"],),
                "analysis_type": ("ANALYSIS_TYPE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_latent_noise"
    CATEGORY = "audio/noise"

    def rand_perlin_2d(self, shape, res, fade=lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]), dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
        
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

    def generate_latent_noise(self, noise_params, width, height, batch_size, noise_type, analysis_type):
        latent_height = height // 8
        latent_width = width // 8
        params = {k: v for k, v in noise_params.items() if k != "timestamps"}
        selected_params = params[noise_type]
        
        intensity = selected_params["intensity"]
        grain = selected_params["grain"]
        persistence = selected_params["persistence"]

        # Adjust parameters based on analysis type
        if analysis_type in ["onset", "segment"]:
            intensity *= 1.5
            grain *= 2.0
        elif analysis_type in ["mel", "spectral"]:
            intensity *= 2.0
            persistence *= 1.2
        elif analysis_type == "tempo":
            grain *= 1.5
            persistence *= 0.8

        noise = torch.zeros((batch_size, 4, latent_height, latent_width), dtype=torch.float32, device="cpu")
        
        if noise_type == "gaussian":
            # Scale noise differently for spectral analysis
            if analysis_type in ["mel", "spectral"]:
                base_noise = torch.randn((batch_size, 4, latent_height, latent_width))
                freq_noise = torch.randn((batch_size, 4, latent_height, latent_width))
                noise = (base_noise + freq_noise * grain) * intensity * persistence
            else:
                noise = torch.randn((batch_size, 4, latent_height, latent_width)) * intensity * persistence
        
        elif noise_type == "salt_pepper":
            # Adjust threshold based on analysis type
            threshold = grain
            if analysis_type in ["onset", "segment"]:
                threshold = min(grain * 1.5, 0.9)
            
            mask = torch.rand((batch_size, 4, latent_height, latent_width)) < threshold
            noise[mask] = intensity
            noise[~mask] = -intensity
            noise *= persistence
            
        elif noise_type == "perlin":
            freq_multiplier = 1.0
            if analysis_type in ["mel", "spectral"]:
                freq_multiplier = 2.0
            elif analysis_type in ["onset", "segment"]:
                freq_multiplier = 0.5
                
            for i in range(batch_size):
                for j in range(4):
                    noise_values = self.rand_perlin_2d(
                        (latent_height, latent_width),
                        (1, 1)
                    )
                    noise_values = intensity * noise_values * persistence
                    if analysis_type in ["mel", "spectral"]:
                        # Add higher frequency detail for spectral analysis
                        detail = self.rand_perlin_2d(
                            (latent_height, latent_width),
                            (2, 2)
                        ) * 0.3
                        noise_values += detail
                    noise[i, j] = noise_values * freq_multiplier

        return ({"samples": noise},)

NODE_CLASS_MAPPINGS = {
    "NoiseToLatentConverter": NoiseToLatentConverter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseToLatentConverter": "Audio Noise to Latent",
}
