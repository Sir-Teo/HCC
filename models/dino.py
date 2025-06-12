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
    # 1. Load teacher checkpoint that was produced by dinov2 training script.
    checkpoint = torch.load(local_weights_path, map_location='cpu')

    # dinov2 "teacher_checkpoint.pth" format is typically {"teacher": state_dict}
    if 'teacher' in checkpoint:
        teacher_state = checkpoint['teacher']
    else:
        # fallback to common keys
        teacher_state = checkpoint.get('model', checkpoint)

    # 2. Inspect checkpoint to infer backbone configuration (embed_dim & patch_size)
    # ---------------------------------------------------------------------------
    def _find_tensor(key_candidates, state):
        for kc in key_candidates:
            if kc in state:
                return state[kc]
        return None

    # Try to locate cls_token and patch_embed to deduce embed_dim and patch_size
    cls_token_tensor = _find_tensor([
        'backbone.cls_token',
        'cls_token'
    ], teacher_state)

    patch_proj_tensor = _find_tensor([
        'backbone.patch_embed.proj.weight',
        'patch_embed.proj.weight'
    ], teacher_state)

    if cls_token_tensor is None or patch_proj_tensor is None:
        raise RuntimeError("Could not infer model architecture from checkpoint. Missing expected keys.")

    embed_dim = cls_token_tensor.shape[-1]
    patch_size = patch_proj_tensor.shape[-1]

    # Derive grid size & image resolution from positional embedding
    pos_embed_tensor = _find_tensor([
        'backbone.pos_embed',
        'pos_embed'
    ], teacher_state)
    if pos_embed_tensor is not None and pos_embed_tensor.ndim == 3:
        num_tokens = pos_embed_tensor.shape[1] - 1  # subtract CLS token
        grid_size = int(num_tokens ** 0.5)
        img_size = grid_size * patch_size
    else:
        img_size = 224  # sensible default

    if embed_dim == 384:
        model_arch = 'dinov2_vits14'
    elif embed_dim == 768:
        model_arch = 'dinov2_vitb14'
    elif embed_dim == 1024:
        model_arch = 'dinov2_vitl14'
    else:
        # fall back to vitg14 for giants (1536+)
        model_arch = 'dinov2_vitg14'

    print(f"[DINOv2] Inferred backbone: embed_dim={embed_dim}, patch_size={patch_size} → hub entry '{model_arch}'.")

    # 3. Instantiate backbone from local dinov2 hub with inferred settings (no pretrained weights)
    base_model = torch.hub.load(
        './dinov2',
        model_arch,
        source='local',
        pretrained=False,
        patch_size=patch_size,  # override default if needed
        img_size=img_size
    )

    # 4. Extract backbone weights only and strip the "backbone." prefix so that
    #    they match keys expected by the hub backbone.
    backbone_prefix = 'backbone.'
    clean_state = {}
    for k, v in teacher_state.items():
        if k.startswith(backbone_prefix):
            clean_key = k[len(backbone_prefix):]  # drop prefix
            clean_state[clean_key] = v

    # Some checkpoints might wrap parameters with "module." if trained with
    # DistributedDataParallel/FSDP. Remove that as well.
    clean_state = {k.replace('module.', ''): v for k, v in clean_state.items()}

    # 5. Finally load the filtered / renamed state dict. We allow non-strict
    #    loading but report any incompatibilities when DEBUG is set.
    incompat = base_model.load_state_dict(clean_state, strict=False)

    if DEBUG:
        if incompat.missing_keys:
            print(f"[DINOv2] WARNING – missing {len(incompat.missing_keys)} keys (showing first 10): {incompat.missing_keys[:10]}")
        if incompat.unexpected_keys:
            print(f"[DINOv2] WARNING – {len(incompat.unexpected_keys)} unexpected keys ignored (showing first 10): {incompat.unexpected_keys[:10]}")

    print("Custom DINOv2 weights loaded successfully and backbone initialised.")
    return DinoV2Wrapper(base_model)