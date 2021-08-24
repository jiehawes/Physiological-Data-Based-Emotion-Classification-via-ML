import torch

shape = (2, 3,)
rand_tensor = torch.rand(shape)
print(f"Size is {rand_tensor.shape}")
print(f"Device Tensor is stored on {rand_tensor.device}")
