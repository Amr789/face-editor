import torch
import numpy as np
import os

class LatentEditor:
    def __init__(self, net, boundary_dir):
        self.net = net
        self.boundary_dir = boundary_dir

    def apply_edit(self, latents, direction_name, strength, target_layers=None):
        path = os.path.join(self.boundary_dir, f"{direction_name}_boundary.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Boundary not found: {path}")
        
        boundary = np.load(path).astype(np.float32)
        edited_latents = latents.clone()
        B, L, D = edited_latents.shape

        # Normalize boundary
        vec = torch.tensor(boundary, dtype=torch.float32, device='cuda')
        
        # Handle different boundary shapes (1D vs 2D) - Simplified logic from your notebook
        if vec.ndim == 1:
            vec = vec / torch.norm(vec)
            vec = vec.view(1, 1, D)
        elif vec.ndim == 2:
            vec = vec / torch.norm(vec, dim=1, keepdim=True)
            # Take the first vector for simplicity or implement per-layer logic if needed
            vec = vec[0].view(1, 1, D) 

        # Apply edit
        if target_layers is None:
            edited_latents += strength * vec
        else:
            for layer_idx in target_layers:
                if layer_idx < L:
                    edited_latents[:, layer_idx, :] += strength * vec.squeeze()

        # Decode
        with torch.no_grad():
            edited_image, _ = self.net.decoder(
                [edited_latents],
                input_is_latent=True,
                randomize_noise=False
            )
        
        return edited_image[0]