from typing import Union, cast
import torch
import matplotlib.pyplot as plt

from rich.progress import track


def W_matrix(eta: torch.Tensor):
    """
    Inputs:
        eta:   [ batch, 3, 1 ]
    Outputs:
        W_eta: [ batch, 3, 3 ]
    """
    batch, _, _ = eta.shape
    phi, theta, _ = eta.squeeze(-1).permute(1, 0)
    zero, one = torch.zeros(batch, device=eta.device), torch.ones(batch, device=eta.device)
    return torch.stack([torch.stack([one, zero, -theta.sin()]),
                        torch.stack([zero, phi.cos(), theta.cos() * phi.sin()]),
                        torch.stack([zero, -phi.sin(), theta.cos() * phi.cos()])]).permute(2, 0, 1)


def R_matrix(eta: torch.Tensor):
    """
    Inputs:
        eta: [ batch, 3, 1 ]
    Outputs:
        R:   [ batch, 3, 3 ]
    """
    phi, theta, psi = eta.squeeze(-1).permute(1, 0)
    ct, st = theta.cos(), theta.sin()
    ch, sh = phi.cos(), phi.sin()
    cp, sp = psi.cos(), psi.sin()
    a = torch.stack([cp * ct, cp * st * sh - sp * ch, cp * st * ch + sp * sh])
    b = torch.stack([sp * ct, sp * st * sh + cp * ch, sp * st * ch - cp * sh])
    c = torch.stack([-st, ct * sh, ct * ch])
    return torch.stack([a, b, c]).permute(2, 0, 1)


def tau_matrix(f: torch.Tensor, M: torch.Tensor, l: Union[float, torch.Tensor]):
    """
    Inputs:
        f:   [ batch, 4, 1 ]
        M:   [ batch, 4, 1 ]
        l:   [ batch ] or float
    Outputs:
        tau: [ batch, 3, 1 ]
    """
    f1, f2, f3, f4 = f.squeeze(-1).permute(1, 0)
    return torch.stack([(f3 - f1) / l,
                        (f2 - f4) / l,
                        M.squeeze(-1).sum(1)]).permute(1, 0)[..., None]


def C_matrix(eta: torch.Tensor, eta_dot: torch.Tensor, I: torch.Tensor):
    """
    Inputs:
        eta:     [ batch, 3, 1 ]
        eta_dot: [ batch, 3, 1 ]
        I: [ batch, 3, 3 ]
    """
    Ixx, Iyy, Izz = torch.diagonal(I, dim1=1, dim2=2).permute(1, 0)
    phi, theta, _ = eta.squeeze(-1).permute(1, 0)
    phi_dot, theta_dot, psi_dot = eta_dot.squeeze(-1).permute(1, 0)
    ct, st = theta.cos(), theta.sin()
    ch, sh = phi.cos(), phi.sin()
    c11 = torch.zeros(eta.shape[0], device=eta.device)
    c12 = ((Iyy - Izz) * (theta_dot * ch * sh + psi_dot * sh**2 * ct)
           + (Izz - Iyy) * (psi_dot * ch**2 * ct)
           - Ixx * psi_dot * ct)
    c13 = (Izz - Iyy) * (psi_dot * ch * sh * ct**2)
    c21 = ((Izz - Iyy) * (theta_dot * ch * sh + psi_dot * sh**2 * ct)
           + (Iyy - Izz) * (psi_dot * ch**2 * ct)
           + Ixx * psi_dot * ct)
    c22 = (Izz - Iyy) * (psi_dot * ch * sh)
    c23 = (- (Ixx * psi_dot * st * ct)
           + (Iyy * psi_dot * sh**2 * ct * st)
           + (Izz * psi_dot * ch**2 * st * ct))
    c31 = (Iyy - Izz) * psi_dot * ch * sh * ct**2 - Ixx * theta_dot * ct
    c32 = ((Izz - Iyy) * (theta_dot * ch * sh * st + phi_dot * sh**2 * ct)
           + (Iyy - Izz) * phi_dot * ch**2 * ct
           + (Ixx * psi_dot - Iyy * psi_dot * sh**2 - Izz * psi_dot * ch**2) * ct * st)
    c33 = ((Iyy - Izz) * phi_dot * ch * sh * ct**2
           + (- Iyy * theta_dot * sh**2 - Izz * theta_dot * ch**2 + Ixx * theta_dot) * ct * st)
    a = torch.stack([c11, c12, c13])
    b = torch.stack([c21, c22, c23])
    c = torch.stack([c31, c32, c33])
    return torch.stack([a, b, c]).permute(2, 0, 1)


if __name__ == "__main__":

    batch = 1

    m = 5.0
    g = 9.81
    l = 2.0

    eta = torch.zeros(batch, 3, 1)
    eta_dot = torch.zeros(batch, 3, 1)
    zi = torch.zeros(batch, 3, 1)
    zi_dot = torch.zeros(batch, 3, 1)

    I = torch.eye(3)[None].repeat(batch, 1, 1)

    t_0, t_f, t_n = 0, 10, 1000
    T = torch.linspace(t_0, t_f, t_n)
    dt = (t_f - t_0) / t_n

    etas = torch.zeros(t_n, *eta.shape)
    zis = torch.zeros(t_n, *zi.shape)
    Rs = torch.zeros(t_n, batch, 3, 3)

    for i, t in track(enumerate(T), total=t_n):

        W = W_matrix(eta)
        J = W.transpose(1, 2) @ I @ W
        R = R_matrix(eta)

        f = torch.rand(batch, 4, 1) * 10
        F = torch.zeros(batch, 3, 1)
        F[:, -1, :] = f.sum(1)

        M = torch.zeros(batch, 4, 1)
        # M = torch.rand(batch, 4, 1) * 1e-2

        eta_dd = J.inverse() @ (tau_matrix(f, M, l) - C_matrix(eta, eta_dot, I) @ eta_dot)
        zi_dd = (1 / m) * (R @ F - m * g * torch.tensor([0, 0, 1]).view(1, 3, 1))

        eta_dot += dt * eta_dd
        zi_dot += dt * zi_dd

        eta += dt * eta_dot
        zi += dt * zi_dot

        etas[i] = eta
        zis[i] = zi

        Rs[i] = R

    fig = plt.figure(figsize=(16, 10))
    (ax_etas, ax_zis) = fig.subplots(nrows=1, ncols=2)
    ax_etas.plot(etas.squeeze())
    ax_etas.grid()
    ax_zis.plot(zis.squeeze())
    ax_zis.grid()
    plt.show()

    def axes(ax: plt.Axes, origin: torch.Tensor, basis: torch.Tensor):
        ax.scatter(*origin, color='orange')
        origin = origin[:, None].expand(3, 3)
        axes = torch.stack([origin, origin + (basis * 25)], dim=-1).permute(1, 0, 2)
        ax.plot(*axes[0], color='blue')
        ax.plot(*axes[1], color='green')
        ax.plot(*axes[2], color='red')

    fig = plt.figure(figsize=(10, 10))
    ax = cast(plt.Axes, fig.add_subplot(projection='3d'))
    ax.invert_yaxis()

    zis = zis.permute(1, 2, 0, 3).squeeze(-1)
    etas = etas.permute(1, 2, 0, 3).squeeze(-1)

    # for zi, eta in zip(zis, etas):
    for b in range(batch):

        # Plot trajectory curve
        x, y, z = zis[b]
        ax.plot(x, y, z)

        for i in range(0, t_n, 200):
            rotation = Rs[i, b].transpose(0, 1).squeeze(0)
            origin = zis[b, :, i]
            axes(ax, origin, rotation)

    ax.scatter(*zis[:, :, 0].permute(1, 0), color='green')
    ax.scatter(*zis[:, :, -1].permute(1, 0), color='red')
    ax.set_aspect('equal')

    plt.show()
