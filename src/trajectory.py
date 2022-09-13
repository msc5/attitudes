
import numpy as np
import matplotlib.pyplot as plt

from quaternion import quaternion as q

if __name__ == "__main__":

    from rich import print

    c = 100
    a = 20
    b = 10
    plot_range = [c + a, -(c + a)]

    n = 10000

    n_halos = 20
    t = np.linspace(0, 2 * np.pi, n)
    s = np.linspace(0, 2 * np.pi, n) * n_halos

    # print(t.shape, s.shape)

    h = c * np.array([np.cos(t), np.sin(t), np.zeros(n)])
    p = np.array([a * np.cos(s), np.zeros(n), b * np.sin(s)])
    q_IH = q.from_theta(np.array([0, 0, 1]), np.pi / 2 - t)
    A_IH = np.array([
        [np.sin(t), np.cos(t), np.zeros(n)],
        [-np.cos(t), np.sin(t), np.zeros(n)],
        [np.zeros(n), np.zeros(n), np.ones(n)]
    ])

    print(h.shape, p.shape, (h + p).shape)
    print(q_IH.shape)

    r = np.zeros((3, n))
    for i in range(n):
        r[:, i] = h[:, i] + A_IH[:, :, i] @ p[:, i]
        # r[:, i] = h[:, i] + q.A(q_IH[:, i]) @ p[:, i]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.plot3D(*r)
    ax.set_xlim3d(plot_range)
    ax.set_ylim3d(plot_range)
    ax.set_zlim3d(plot_range)
    plt.show()
