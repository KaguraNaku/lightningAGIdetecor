"""Model parameters for a fresh Lightning fit."""

BACKBONE_ID = "openai/clip-vit-large-patch14"
BACKBONE_REVISION = None  # Optional Hugging Face commit, tag, or branch.
LAYERS = (0, 3, 7, 10, 13, 16, 20, 23)  # Zero-based transformer blocks.
PROJECTION_RANK = 128  # Projected width per selected block.
RIDGE_LAMBDA = 0.15  # Shared by both readout branches.
