import torch

# Compute thrust given current inputs and thrust coefficient.


def thrust(inputs, k):
    # Inputs are values for ${\omega_i}^2$
    return torch.tensor([[0, 0, k * sum(inputs)]])

# Compute torques, given current inputs, length, drag coefficient, and thrust coefficient.


def torques(inputs, L, b, k):
    # Inputs are values for ${\omega_i}^2$
    return torch.tensor([L * k * (inputs(1) - inputs(3)),
                         L * k * (inputs(2) - inputs(4)),
                         b * (inputs(1) - inputs(2) + inputs(3) - inputs(4))])


def acceleration(inputs, angles, xdot, m, g, k, kd):
    gravity = torch.tensor([[0, 0, - g]])
    R = rotation(angles)
    T = R * thrust(inputs, k)
    Fd = -kd * xdot
    a = gravity + 1 / m * T + Fd
    return a


def angular_acceleration(inputs, omega, I, L, b, k):
    tau = torques(inputs, L, b, k)
    omegaddot = torch.inverse(I) * (tau - torch.cross(omega, I * omega))
    return omegaddot


if __name__ == "__main__":

    # Simulation times, in seconds.
    start_time = 0
    end_time = 10
    dt = 0.005
    # times = start_time:dt:end_time
    times = torch.linspace(start_time, end_time, int((end_time - start_time) / dt))

    # Number of points in the simulation.
    N = times.numel()

    # Initial simulation state.
    x = torch.tensor([[0, 0, 10]])
    xdot = torch.zeros((3, 1))
    theta = torch.zeros((3, 1))

    # Simulate some disturbance in the angular velocity.
    # The magnitude of the deviation is in radians / second.
    deviation = 100
    thetadot = torch.deg2rad(2 * deviation * torch.rand(3, 1) - deviation)

    # Step through the simulation, updating the state.
    for t in times:

        # # Take input from our controller.
        # i = input(t)

        omega = thetadot2omega(thetadot, theta)

        # Compute linear and angular accelerations.
        a = acceleration(i, theta, xdot, m, g, k, kd)
        omegadot = angular_acceleration(i, omega, I, L, b, k)

        omega = omega + dt * omegadot
        thetadot = omega2thetadot(omega, theta)
        theta = theta + dt * thetadot
        xdot = xdot + dt * a
        x = x + dt * xdot
