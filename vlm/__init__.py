"""Vision-Language Model integration (CLIP encoder, prompt generation)."""

from .clip_encoder import CLIPEncoder
from .feature_fusion import ViewAwarePromptGenerator, FeatureFusion

__all__ = ["CLIPEncoder", "ViewAwarePromptGenerator", "FeatureFusion"]
