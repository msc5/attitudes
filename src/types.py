import torch
from typing import Any, Union
from torchtyping import TensorType, patch_typeguard


Float = TensorType[1, torch.float]

Vector3 = TensorType[..., Any]
Matrix = TensorType[..., Any, Any]

Quaternion = TensorType[..., 4]

patch_typeguard()
