"""Model adapters for the forecasting benchmark."""

from __future__ import annotations

from benchmark.models.base import BaseForecaster


def _get_timesfm3_class():
    from benchmark.models.timesfm3 import TimesFM3Forecaster
    return TimesFM3Forecaster


def _get_prophet_class():
    from benchmark.models.prophet import ProphetForecaster
    return ProphetForecaster


def _get_lightgbm_class():
    from benchmark.models.lightgbm import LightGBMForecaster
    return LightGBMForecaster


def _get_ets_class():
    from benchmark.models.ets import ETSForecaster
    return ETSForecaster


def _get_naive_class():
    from benchmark.models.naive import NaiveForecaster
    return NaiveForecaster


# Model registry - lazy imports to avoid importing all dependencies at once
# Each entry maps: key -> (getter_function, display_name)
_MODEL_REGISTRY = {
    "timesfm3": (_get_timesfm3_class, "TimesFM 3.0"),
    "prophet": (_get_prophet_class, "Prophet"),
    "lightgbm": (_get_lightgbm_class, "LightGBM"),
    "ets": (_get_ets_class, "ETS (Holt-Winters)"),
    "naive": (_get_naive_class, "Naive (s=12)"),
}


def get_model_class(name: str) -> type[BaseForecaster]:
    """Get a model class by name, importing only when needed.
    
    Args:
        name: Model key (e.g., "timesfm3", "prophet", "lightgbm").
    
    Returns:
        The model class.
    
    Raises:
        KeyError: If model name not found.
        ImportError: If model dependencies not installed.
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    
    getter, _ = _MODEL_REGISTRY[name]
    return getter()


def list_models() -> list[str]:
    """List all available model keys."""
    return list(_MODEL_REGISTRY.keys())


# For backward compatibility - lazy ALL_MODELS proxy
class _LazyModelsDict:
    """Dict-like access that imports models on demand."""
    
    def __getitem__(self, key: str) -> type[BaseForecaster]:
        return get_model_class(key)
    
    def __contains__(self, key: str) -> bool:
        return key in _MODEL_REGISTRY
    
    def keys(self):
        return _MODEL_REGISTRY.keys()
    
    def values(self):
        return [get_model_class(k) for k in _MODEL_REGISTRY.keys()]
    
    def items(self):
        return [(k, get_model_class(k)) for k in _MODEL_REGISTRY.keys()]
    
    def __iter__(self):
        return iter(_MODEL_REGISTRY.keys())
    
    def __len__(self):
        return len(_MODEL_REGISTRY)
    
    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, ImportError):
            return default


ALL_MODELS = _LazyModelsDict()

__all__ = ["BaseForecaster", "ALL_MODELS", "get_model_class", "list_models"]
