import torch

from .cli import console

from . import types as tt


def cross(v: tt.Vector3) -> tt.Matrix:
    v1, v2, v3 = v[..., 0], v[..., 1], v[..., 2]
    zeros = torch.zeros(v1.shape)
    cross = [[zeros, -v3, v2],
             [v3, zeros, -v1],
             [-v2, v1, zeros]]
    cross = [torch.stack(row, dim=-1) for row in cross]
    cross = torch.stack(cross, dim=-1)
    cross = cross.transpose(-1, -2)
    return cross


def eye(*shape: int) -> tt.Matrix:
    breakpoint()
    eye = torch.eye(shape[-1])
    eye = eye[[None] * (len(shape) - 1)]
    eye = eye.repeat(*shape[:-1], 1, 1)
    return eye


if __name__ == "__main__":

    p = torch.rand(1, 2, 3)
    q = torch.rand(1, 2, 3)

    console.log(p)
    console.log(q)

    c = cross(p)
    console.log(c)

    console.log(c @ p[..., None])

    e = eye(*p.shape)
