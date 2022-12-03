from typing import Any
import torch


def hat(x: torch.Tensor):
    """
    Inputs:
        x:     [ batch, 3, 1 ]
    Outputs:
        x_hat: [ batch, 3, 3 ]
    """
    x1, x2, x3 = x.squeeze(-1).permute(1, 0)
    zero = torch.zeros(x.shape[0], device=x.device)
    a = torch.stack([zero, -x3, x2])
    b = torch.stack([x3, zero, -x1])
    c = torch.stack([-x2, x1, zero])
    return torch.stack([a, b, c]).permute(2, 0, 1)


def unhat(x: torch.Tensor):
    """
    Inputs:
        x_hat: [ batch, 3, 3 ]
    Outputs:
        x:     [ batch, 3, 1 ]
    """
    x1 = x[:, 2, 1]
    x2 = x[:, 0, 2]
    x3 = x[:, 1, 0]
    return torch.stack([x1, x2, x3]).view(x.shape[0], 3, 1)


def rotation_matrix(c: torch.Tensor, angle: Any):
    """
    Inputs:
        c: [ batch, 3, 3 ]
    Outputs:
        R: [ batch, 3, 3 ]
    """
    return torch.matrix_exp(angle * hat(c))


def euler_matrix(angles: torch.Tensor):
    """
    Inputs:
        angles: [ batch, 3, 1 ]
    Outputs:
        euler:  [ batch, 3, 3 ]
    """
    x = torch.eye(3, device=angles.device).view(3, 3, 1)
    c = angles[..., None] * hat(x)[None]
    c = c.contiguous()
    R = torch.matrix_exp(c.view(-1, 3, 3)).view(angles.shape[0], -1, 3, 3)
    # return R[:, 2] @ R[:, 1] @ R[:, 0] # z y x
    return R[:, 0] @ R[:, 1] @ R[:, 2]  # x y z


if __name__ == "__main__":

    from src.cli import console

    x = torch.eye(3).view(3, 3, 1)

    hat_x = hat(x)
    rotation = rotation_matrix(x, torch.pi / 2)
    unhat_x = unhat(hat_x)

    console.print(x)
    console.print('hat_x')
    console.print(hat_x)
    console.print('Rotations')
    console.print(rotation)
    console.print(unhat_x)

    angles = torch.rand(5, 3, 1)
    euler = euler_matrix(angles)

    from .dynamics import R

    test_euler = R(angles)

    console.print(euler)
    console.print(test_euler)
    console.print((euler - test_euler).abs())
