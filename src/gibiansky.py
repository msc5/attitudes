import matplotlib.pyplot as plt
import torch


def rotation(angles):
    # Compute rotation matrix for a set of angles.
    phi = angles[2]
    theta = angles[1]
    psi = angles[0]

    R = torch.zeros(3, 3)
    R[:, 0] = torch.tensor([
        [phi.cos() * theta.cos()],
        [theta.cos() * phi.sin()],
        [- theta.sin()],
    ]).squeeze()
    R[:, 1] = torch.tensor([
        [phi.cos() * theta.sin() * psi.sin() - psi.cos() * phi.sin()],
        [phi.cos() * psi.cos() + phi.sin() * theta.sin() * psi.sin()],
        [theta.cos() * psi.sin()]
    ]).squeeze()
    R[:, 2] = torch.tensor([
        [phi.sin() * psi.sin() + phi.cos() * psi.cos() * theta.sin()],
        [psi.cos() * phi.sin() * theta.sin() - phi.cos() * psi.sin()],
        [theta.cos() * psi.cos()]
    ]).squeeze()
    return R


def thrust(inputs, k):
    # Compute thrust given current inputs and thrust coefficient.
    # Inputs are values for ${\omega_i}^2$
    return torch.tensor([[0, 0, k * sum(inputs)]])


def torques(inputs, L, b, k):
    # Compute torques, given current inputs, length, drag coefficient, and thrust coefficient.
    # Inputs are values for ${\omega_i}^2$
    return torch.tensor([L * k * (inputs[0] - inputs[2]),
                         L * k * (inputs[1] - inputs[3]),
                         b * (inputs[0] - inputs[1] + inputs[2] - inputs[3])])


def acceleration(inputs, angles, xdot, m, g, k, kd):
    gravity = torch.tensor([[0, 0, - g]]).T
    R = rotation(angles)
    T = R @ thrust(inputs, k).T
    Fd = -kd * xdot
    a = gravity + 1 / m * T + Fd
    return a


def angular_acceleration(inputs, omega, I, L, b, k):
    tau = torques(inputs, L, b, k)
    omegaddot = torch.inverse(I) @ (tau[:, None] - torch.cross(omega, I @ omega))
    return omegaddot


def thetadot2omega(thetadot, angles):
    # Convert derivatives of roll, pitch, yaw to omega.
    phi = angles[2]
    theta = angles[1]
    # psi = angles[0]
    W = torch.tensor([
        [1, 0, -theta.sin()],
        [0, phi.cos(), theta.cos() * phi.sin()],
        [0, -phi.sin(), theta.cos() * phi.cos()],
    ])
    omega = W @ thetadot
    return omega


def omega2thetadot(omega, angles):
    # Convert omega to roll, pitch, yaw derivatives
    phi = angles[2]
    theta = angles[1]
    # psi = angles[0]
    W = torch.tensor([
        [1, 0, -theta.sin()],
        [0, phi.cos(), theta.cos() * phi.sin()],
        [0, -phi.sin(), theta.cos() * phi.cos()],
    ])
    thetadot = torch.inverse(W) @ omega
    return thetadot


if __name__ == "__main__":

    ax = plt.axes(projection='3d')

    for _ in range(1):

        # Physical constants.
        g = 9.81
        m = 0.5
        L = 0.25
        k = 3e-6
        b = 1e-7
        I = torch.diag(torch.tensor([5e-3, 5e-3, 10e-3]))
        kd = 0.25

        # Simulation times, in seconds.
        start_time = 0
        end_time = 10
        dt = 0.005
        # times = start_time:dt:end_time
        times = torch.linspace(start_time, end_time, int((end_time - start_time) / dt))

        # Number of points in the simulation.
        N = times.numel()

        # Initial simulation state.
        x = torch.tensor([[0, 0, 100]]).T
        xdot = torch.zeros((3, 1))
        theta = torch.zeros((3, 1))

        # Simulate some disturbance in the angular velocity.
        # The magnitude of the deviation is in radians / second.
        deviation = 100
        thetadot = torch.deg2rad(2 * deviation * torch.rand(3, 1) - deviation)
        # thetadot = torch.zeros(3, 1)

        # input = torch.rand(4, N)
        # inputs = torch.rand(4, N)
        inputs = torch.ones(4, N) * 10

        outputs = torch.zeros(N, 3)

        # Step through the simulation, updating the state.
        for ind, t in enumerate(times):

            # Take input from our controller.
            i = inputs[:, ind]

            omega = thetadot2omega(thetadot, theta)

            # Compute linear and angular accelerations.
            a = acceleration(i, theta, xdot, m, g, k, kd)
            omegadot = angular_acceleration(i, omega, I, L, b, k)

            omega = omega + dt * omegadot
            thetadot = omega2thetadot(omega, theta)
            theta = theta + dt * thetadot
            xdot = xdot + dt * a
            x = x + dt * xdot

            outputs[ind] = x.squeeze()

        ax.scatter(outputs[:, 0], outputs[:, 1], outputs[:, 2], c=-outputs[:, 2])

    plt.show()
