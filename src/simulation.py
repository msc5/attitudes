import os
from pathlib import Path
import torch

import json
import time
from rich.progress import track
from rich.console import Console

from . import plots as pp
from .dynamics import W_matrix, R_matrix, tau_matrix, C_matrix
import argparse


parser = argparse.ArgumentParser(prog='Cleanup', description='Clean up results')

# Optional
parser.add_argument('--tag', type=str, default='', help='Tag to name files')

if __name__ == "__main__":

    console = Console()

    args = parser.parse_args().__dict__
    console.log(args)

    batch = 1

    m = 0.5
    g = 9.81
    l = 0.25

    # k = 3e-6
    # b = 1e-7
    k = 1
    b = 1

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    # Initial states
    # eta = torch.zeros(batch, 3, 1, device=device)
    # eta_dot = torch.zeros(batch, 3, 1, device=device)
    eta = torch.normal(0, 0.1, size=(batch, 3, 1), device=device)
    eta_dot = torch.normal(0, 0.1, size=(batch, 3, 1), device=device)

    zi = torch.zeros(batch, 3, 1, device=device)
    zi_dot = torch.zeros(batch, 3, 1, device=device)

    eta_int = torch.zeros(batch, 3, 1, device=device)
    ez = torch.tensor([0, 0, 1], device=device).view(1, 3, 1)

    I = torch.eye(3, device=device)[None].repeat(batch, 1, 1)

    t_0, t_f, t_n = 0, 10.0, 5000
    T = torch.linspace(t_0, t_f, t_n, device=device)
    dt = (t_f - t_0) / t_n

    etas = torch.zeros(t_n, *eta.shape, device=device)
    etas_dot = torch.zeros(t_n, *eta.shape, device=device)
    zis = torch.zeros(t_n, *zi.shape, device=device)
    Rs = torch.zeros(t_n, batch, 3, 3, device=device)
    fs = torch.zeros(t_n, batch, 4, 1, device=device)

    console.print(f'Simulating {batch} Quadcopters...')
    start = time.perf_counter()

    kd = 4
    kp = 8
    ki = 5.5

    results = {'I': I,
               'kd': kd, 'kp': kp, 'ki': ki,
               't_0': t_0, 't_f': t_f, 't_n': t_n, 'T': T,
               'm': m, 'g': g, 'l': l}

    for i, t in track(enumerate(T), total=t_n):

        W = W_matrix(eta)
        J = W.transpose(1, 2) @ I @ W
        R = R_matrix(eta)

        # Controller
        if i > (0):
            u = - (kd * eta_dot + kp * eta + ki * eta_int)
            j = J @ u
            j1, j2, j3 = j[:, 0], j[:, 1], j[:, 2]
            T = (m * g) / (eta[:, 0].cos() * eta[:, 1].cos())
            f1 = j1 / (2 * k * l) + j3 / (4 * b)
            f2 = j2 / (2 * k * l) - j3 / (4 * b)
            f3 = - j1 / (2 * k * l) + j3 / (4 * b)
            f4 = - j2 / (2 * k * l) - j3 / (4 * b)
            f = (T / (4 * k)) + torch.stack([f1, f2, f3, f4], dim=1)
            # f = f + torch.normal(0, 0.1, size=(batch, 4, 1), device=device)
        else:
            # f = torch.ones(batch, 4, 1, device=device) * ((m * g + 1) / 4)
            f = torch.ones(batch, 4, 1, device=device) * ((m * g + 1) / 4)
            f[..., 2:, 0] = f[..., 2:, 0] + 1

        F = torch.zeros(batch, 3, 1, device=device)
        F[:, -1, :] = f.sum(1)

        # Equations of Motion
        eta_dd = J.inverse() @ (tau_matrix(f, l, k, b) - C_matrix(eta, eta_dot, I) @ eta_dot)
        zi_dd = (1 / m) * (R @ F - m * g * ez)

        # Integration
        eta_dot += dt * eta_dd
        zi_dot += dt * zi_dd
        eta += dt * eta_dot
        zi += dt * zi_dot
        eta_int += dt * eta

        etas[i] = eta
        etas_dot[i] = eta_dot
        zis[i] = zi
        Rs[i] = R

    stop = time.perf_counter()
    console.print(f'Simulation Completed in {stop - start:5.3f} s')

    Rs = Rs.permute(1, 0, 2, 3)
    zis = zis.permute(1, 0, 2, 3).squeeze(-1)
    etas = etas.permute(1, 0, 2, 3).squeeze(-1)
    etas_dot = etas_dot.permute(1, 0, 2, 3).squeeze(-1)

    results = {**results, 'etas': etas, 'etas_dot': etas_dot, 'zis': zis, 'Rs': Rs}

    dir = Path('results')
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Load metadata and get current run
    meta_path = dir.joinpath('metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as stream:
            meta = json.load(stream)
    else:
        meta = {'run': 0}
    run = meta['run'] = meta['run'] + 1
    with open(meta_path, 'w') as stream:
        json.dump(meta, stream)

    # Save simulation results
    run_name = f'{run:03d}' if args['tag'] == '' else f'{run:03d}-{args["tag"]}'
    run_path = dir.joinpath(run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    pp.traj3d(run_path.joinpath('traj3d'), results)
    pp.traj2d(run_path.joinpath('traj2d'), results)
    torch.save(results, run_path.joinpath('results.pt'))
