import numpy as np
from PIL import Image

class NoiseToLatentConverter:
    """Converts audio-driven noise parameters to latent noise format compatible with VAE nodes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_params": ("NOISE_PARAMS",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "noise_type": (["gaussian", "salt_pepper", "perlin"],),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_latent_noise"
    CATEGORY = "audio/noise"

    def generate_latent_noise(self, noise_params, width, height, batch_size, noise_type):
        # Get parameters for the selected noise type
        params = noise_params[noise_type]
        
        # Calculate latent dimensions (1/8 of image dimensions for VAE)
        latent_width = width // 8
        latent_height = height // 8
        
        # Generate base noise
        if noise_type == "gaussian":
            noise = np.random.normal(0, params.intensity, 
                                   (batch_size, 4, latent_height, latent_width))
        elif noise_type == "salt_pepper":
            noise = np.random.choice([-params.intensity, params.intensity],
                                   size=(batch_size, 4, latent_height, latent_width),
                                   p=[1-params.grain, params.grain])
        else:  # perlin
            noise = self._generate_perlin_noise(latent_height, latent_width, 
                                              params.grain, params.persistence,
                                              batch_size)
        
        return ({"samples": noise},)
    
    def _generate_perlin_noise(self, height, width, scale, persistence, batch_size):
        from opensimplex import OpenSimplex
        
        noise = np.zeros((batch_size, 4, height, width))
        for b in range(batch_size):
            for c in range(4):
                simplex = OpenSimplex(seed=np.random.randint(0, 1000000))
                for y in range(height):
                    for x in range(width):
                        noise[b,c,y,x] = simplex.noise2(x*scale, y*scale)
        
        return noise

class NoiseVisualizer:
    """Visualizes the generated noise as an image"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_params": ("NOISE_PARAMS",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "noise_type": (["gaussian", "salt_pepper", "perlin"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_noise"
    CATEGORY = "audio/noise"

    def visualize_noise(self, noise_params, width, height, noise_type):
        # Get parameters for the selected noise type
        params = noise_params[noise_type]
        
        # Generate noise based on type
        if noise_type == "gaussian":
            noise = np.random.normal(0.5, params.intensity/2, (height, width))
            noise = np.clip(noise, 0, 1)
        
        elif noise_type == "salt_pepper":
            noise = np.random.choice([0, 1], size=(height, width),
                                   p=[1-params.grain, params.grain])
        
        else:  # perlin
            noise = self._generate_perlin_noise(height, width, 
                                              params.grain, params.persistence)
            noise = (noise + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Convert to PIL Image format
        noise_image = (noise * 255).astype(np.uint8)
        image = Image.fromarray(noise_image)
        
        # Convert to RGB and stack for ComfyUI compatibility
        rgb_image = np.array(image.convert('RGB'))
        return (rgb_image[None, ...],)
    
    def _generate_perlin_noise(self, height, width, scale, persistence):
        from opensimplex import OpenSimplex
        
        simplex = OpenSimplex(seed=np.random.randint(0, 1000000))
        noise = np.zeros((height, width))
        
        for y in range(height):
            for x in range(width):
                noise[y,x] = simplex.noise2(x*scale, y*scale)
        
        return noise

# Register the nodes
NODE_CLASS_MAPPINGS = {
    "NoiseToLatentConverter": NoiseToLatentConverter,
    "NoiseVisualizer": NoiseVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoiseToLatentConverter": "Audio Noise to Latent",
    "NoiseVisualizer": "Visualize Audio Noise"
}
