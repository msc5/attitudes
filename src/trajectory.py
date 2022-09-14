import torch
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d as _

from . import quaternion as Q
from . import vector as V


if __name__ == "__main__":

    c = 100     # Speed of major-axis rotation
    a = 20      # Minor-axis radius
    b = 20      # Major-axis radius
    plot_range = [c + a, -(c + a)]

    n = 10000

    n_halos = 20
    t = torch.linspace(0, 2 * torch.pi, n)
    s = torch.linspace(0, 2 * torch.pi, n) * n_halos

    zero = torch.zeros(n)
    one = torch.ones(n)

    h = c * torch.stack([t.cos(), t.sin(), zero], dim=-1)
    p = torch.stack([a * s.cos(), zero, b * s.sin()], dim=-1)

    e = V.batch(torch.tensor([0, 0, 1]), [n])
    q_IH = Q.from_theta(e, (torch.pi / 2) - t)

    r = h + (Q.A(q_IH) @ p[..., None]).squeeze()

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    r = r.numpy()
    ax.plot3D(r[:, 0], r[:, 1], r[:, 2])
    ax.set_xlim3d(plot_range)
    ax.set_ylim3d(plot_range)
    ax.set_zlim3d(plot_range)
    plt.show()
