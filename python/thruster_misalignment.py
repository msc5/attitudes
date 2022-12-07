
import numpy as np

from rich import print

from numpy.linalg import norm


def format_var(name: str, var, unit: str = ''):
    if isinstance(var, np.ndarray):
        x, y, z = var
        formatted_var = f'[{x: 7.5f}, {y: 7.5f}, {z: 7.5f}]'
    if isinstance(var, int) or isinstance(var, float):
        formatted_var = f'{var:<7.5f}'
    if isinstance(var, str):
        formatted_var = var
    print(f'{name:>35} : {formatted_var} {unit}')


def inch_to_meter(inch):
    return 0.0254 * inch


def lb_to_kg(lb):
    return 0.453592 * lb


def torque(t, c_o, c_f, F, offset):
    # Initial COM-to-thruster moment arm (Also thrust vector)
    t_o = t - c_o
    l_o = norm(t_o)
    # Final COM-to-thruster moment arm
    t_f = t - c_f
    l_f = norm(t_f)
    # Final angle between thrust vector and final moment arm
    theta = np.arccos(np.dot(t_o, t_f) / (l_o * l_f)) + offset * (np.pi / 180)
    # Final instantaneous torque on spacecraft COM
    T = F * l_f * np.sin(theta)
    # Total Momentum buildup over whole maneuver
    H = (1 / 2) * time_burn * T
    return theta, t_o, t_f, T, H


if __name__ == "__main__":

    def from_doc(x): return inch_to_meter(np.array(x))

    # Centers of gravity (c_o, c_f) and thruster position (t)
    c_o = from_doc([5.3279459e+01, 3.0405055e+01, 2.5733602e+01])
    c_f = from_doc([5.3977153e+01, 2.9710242e+01, 2.3760950e+01])
    t = c_o * np.array([0, 1, 1])

    F = 425                 # Nominal Thruster Force
    time_burn = 3492        # Burn Time (s)
    misalign = 0.5          # Thruster misalignment (deg)

    theta, t_o, t_f, T, H = torque(t, c_o, c_f, F, misalign)

    format_var('Theta', theta * (180 / np.pi), 'deg')
    format_var('Thrust Vector', t, 'm')
    format_var('t_o', t_o, 'm')
    format_var('t_f', t_f, 'm')
    format_var('c_o', c_o, 'm')
    format_var('c_f', c_f, 'm')
    format_var('Initial Moment Arm', t_o, 'm')
    format_var('Final Moment Arm', t_f, 'm')
    format_var('Final Torque', T, 'Nm')
    format_var('Momentum Accumulation', H, 'Nms')
