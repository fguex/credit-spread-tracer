from .features_builder import build_features_dataset

# Backwards-compatible alias
build_dataset = build_features_dataset

__all__ = ["build_features_dataset", "build_dataset"]
