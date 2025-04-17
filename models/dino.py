# models/dino.py
import torch
from torch import nn

DEBUG = False

class DinoV2Wrapper(nn.Module):
    """
    Wraps the DINOv2 model to return a single feature vector per image.
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.patch_pool = nn.AdaptiveAvgPool1d(1)  # Add patch-level adaptive pooling

    def forward(self, x):
        return self.base_model(x)

    def forward_features(self, x):
        token_embeddings = self.forward(x)
        if token_embeddings.ndim == 3 and token_embeddings.size(1) > 1:
            patch_tokens = token_embeddings[:, 1:, :]  # [batch, patches, embed_dim]
            # Use AdaptiveAvgPool1d to pool patch tokens
            pooled = self.patch_pool(patch_tokens.transpose(1, 2))  # [batch, embed_dim, 1]
            features = pooled.squeeze(-1)  # [batch, embed_dim]
            return features
        elif token_embeddings.ndim == 2:
            return token_embeddings
        else:
            raise ValueError(f"Unexpected token_embeddings shape: {token_embeddings.shape}")

def load_dinov2_model(local_weights_path):
    """
    Loads the DINOv2 model using local weights.
    """
    model_arch = "dinov2_vitb14"  # Adjust as needed
    print(f"Loading {model_arch} from local hub...")

    base_model = torch.hub.load(
        './dinov2',
        model_arch,
        source='local'
    )

    checkpoint = torch.load(local_weights_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        base_model.load_state_dict(checkpoint['teacher']['model'])

    print("Local DINOv2 weights loaded successfully.")
    return DinoV2Wrapper(base_model)
