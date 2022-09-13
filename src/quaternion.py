from typing import Union
import torch
import numpy as np

from . import types as tt
from . import vector as V


def tensorize(f: Union[float, tt.Float]) -> tt.Float:
    if not isinstance(f, torch.Tensor):
        return torch.tensor([f])
    else:
        return f


def from_theta(e: tt.Vector3, theta: tt.Float) -> tt.Quaternion:
    return torch.cat([e * (theta / 2).sin(), (theta / 2).cos()], dim=-1)


def psi(q: tt.Quaternion) -> tt.Matrix:
    breakpoint()
    qv = q[..., 0:3]
    q4 = q[..., 3]
    qcross = V.cross(qv)
    a = q4 * V.eye(qcross.shape) - qcross
    b = - qv.transpose(-1, -2)
    return np.concatenate((a, b), axis=0)


def xi(q):
    shape = q.shape[:-1]
    qv = q[..., 0:3]
    q4 = q[..., 3]
    qcross = V.cross(qv)
    a = q[3] * np.eye(3) + qcross
    b = - (qv.T).reshape((1, 3))
    return np.concatenate((a, b), axis=0)


def cross(q):
    return np.append(psi(q), q, axis=1)


def dot(q):
    return np.append(xi(q), q, axis=1)


def A(q):
    return xi(q).T @ psi(q)


if __name__ == "__main__":

    from .cli import console

    v = torch.tensor([0, 0, 1])
    theta = tensorize(torch.pi / 2)

    rot = from_theta(v, theta)
    console.log(rot, rot.shape)

    rot_xi = xi(rot)
    console.log(rot_xi, rot_xi.shape)

    rot_cross = cross(rot)
    console.log(rot_cross, rot_cross.shape)

    rot_A = A(rot)
    console.log(rot_A, rot_A.shape)
