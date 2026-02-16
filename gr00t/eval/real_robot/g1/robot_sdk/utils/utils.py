from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
except ImportError:
    torch = None

def recursive_add_extra_dim(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively add an extra dim to arrays or scalars.

    GR00T Policy Server expects:
        obs: (batch=1, time=1, ...)
    Calling this function twice achieves that.
    """
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs

def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)

def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def _to_numpy_image(image: Any) -> Optional[np.ndarray]:
    if image is None:
        return None
    if hasattr(image, "detach"):
        image = image.detach()
    if hasattr(image, "cpu"):
        image = image.cpu()
    if hasattr(image, "numpy"):
        image = image.numpy()
    return np.asarray(image)


def _action_to_sized_vector(action: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if action.shape[0] == target_len:
        return action.astype(np.float32, copy=False)
    out = np.zeros(target_len, dtype=np.float32)
    copy_len = min(target_len, action.shape[0])
    out[:copy_len] = action[:copy_len]
    return out
