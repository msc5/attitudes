from typing import Union, cast
import torch


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


def tau_matrix(f: torch.Tensor, l: Union[float, torch.Tensor], k: Union[float, torch.Tensor], b: Union[float, torch.Tensor]):
    """
    Inputs:
        f:   [ batch, 4, 1 ]
        M:   [ batch, 4, 1 ]
        l:   [ batch ] or float
    Outputs:
        tau: [ batch, 3, 1 ]
    """
    f1, f2, f3, f4 = (f**2).squeeze(-1).permute(1, 0)
    return torch.stack([(f3 - f1) * k * l,
                        (f2 - f4) * k * l,
                        (f1 - f2 + f3 - f4) * b]).permute(1, 0)[..., None]


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
