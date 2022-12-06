import torch
import matplotlib.pyplot as plt
from typing import Union, cast

import warnings
warnings.filterwarnings("ignore")


def traj2d(path: str, results: dict):

    zis, etas, etas_dot, T = results['zis'], results['etas'], results['etas_dot'], results['T']
    batch = len(zis)

    fig = plt.figure(figsize=(16, batch * 5), layout='tight')
    axes = fig.subplots(nrows=batch, ncols=3, squeeze=False)

    for b in range(len(zis)):

        axes[b, 0].plot(T, zis[b])
        axes[b, 0].grid()
        axes[b, 0].set_title(r'Simulation $\xi$ vs. Time')
        axes[b, 0].set_ylabel('Position (m)')
        axes[b, 0].set_xlabel('Time (s)')
        axes[b, 0].legend(['x', 'y', 'z'])

        axes[b, 1].plot(T, etas[b])
        axes[b, 1].grid()
        axes[b, 1].set_title(r'Simulation $\eta$ vs. Time')
        axes[b, 1].set_ylabel('Angle (rad)')
        axes[b, 1].set_xlabel('Time (s)')
        axes[b, 1].legend(['ϕ', 'θ', 'ψ'])

        axes[b, 2].plot(T, etas_dot[b])
        axes[b, 2].grid()
        axes[b, 2].set_title('Simulation $\dot{\eta}$ vs. Time')
        axes[b, 2].set_ylabel('Angle (rad)')
        axes[b, 2].set_xlabel('Time (s)')
        axes[b, 2].legend([r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$'])

    plt.show()

    fig.savefig(path)
    plt.close()


def axes(ax: plt.Axes, origin: torch.Tensor, basis: torch.Tensor):
    ax.scatter(*origin, color='orange')
    origin = origin[:, None].expand(3, 3)
    axes = torch.stack([origin, origin + (basis * 25)], dim=-1).permute(1, 0, 2)
    ax.plot(*axes[0], color='blue')
    ax.plot(*axes[1], color='green')
    ax.plot(*axes[2], color='red')


def tracers(zis: torch.Tensor, rotation: torch.Tensor, l: Union[float, torch.Tensor]):
    """
    Inputs:
        zis:  [ batch, 3, t_n ]
        etas: [ batch, 3, t_n ]
    Outputs:
        tracer: [ batch, 3, 4, t_n ]
    """
    props = torch.tensor([[1, 0, 0], [0, 1, 0]], device=zis.device).permute(1, 0)
    props = l * torch.cat([props, -props], dim=-1)
    tracer = zis[..., None] + rotation @ props[None]
    tracer = tracer.permute(2, 1, 0)
    return tracer


def traj3d(path: str, results: dict):

    zis, etas, Rs = results['zis'], results['etas'], results['Rs']
    l, T = results['l'], results['T']

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Quadrotor Trajectory')
    ax = cast(plt.Axes, fig.add_subplot(projection='3d'))
    ax.invert_yaxis()

    batch = len(zis)

    for b in range(batch):

        tracer = tracers(zis[b], Rs[b], l)
        for i, trace in enumerate(tracer):
            # ax.scatter(*trace.cpu(), c=plt.cm.get_cmap('rainbow')((b / batch) + i * 1e-2))
            ax.scatter(*trace.cpu(), marker='.', c=plt.cm.get_cmap('cool')(T / results['t_f']))
            # ax.scatter(*trace[:, 0].cpu(), color='green')
            # ax.scatter(*trace[:, -1].cpu(), color='red')

    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    fig.subplots_adjust(top=1, bottom=0, left=0, right=0.95)
    plt.show()

    fig.savefig(path)
    plt.close()
