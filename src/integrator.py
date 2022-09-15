from typing import Literal, Optional
import time
import matplotlib.pyplot as plt
import torch

from rich import print


class Integrator:

    @classmethod
    def methods(cls, A: torch.Tensor, dt: float):

        # TODO : implement nonlinear integration

        def rk4(x: torch.Tensor, t: float):
            k1 = A @ x
            k2 = A @ (x + dt * (k1 / 2))
            k3 = A @ (x + dt * (k2 / 2))
            k4 = A @ (x + dt * k3)
            x = x + (1 / 6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
            return x, t

        def euler(x: torch.Tensor, t: float):
            x = x + dt * (A @ x)
            t = t + dt
            return x, t

        return {'euler': euler,
                'rk4': rk4}

    @classmethod
    def integrate(cls,
                  dynamics: torch.Tensor,
                  initial: Optional[torch.Tensor] = None,
                  method: Literal['euler', 'rk4'] = 'euler',
                  dt: float = 0.1,
                  steps: int = 1000):

        if initial is None:
            initial = torch.zeros(len(dynamics))
        assert isinstance(initial, torch.Tensor)

        assert len(dynamics.shape) == 2
        assert dynamics.shape[0] == dynamics.shape[1]

        A = dynamics.to(torch.float32)
        if len(initial.shape) == 2:
            A, initial = A[None], initial[:, :, None]
        X = torch.zeros(steps, *initial.shape)
        T = torch.zeros(steps)

        integrator = cls.methods(A, dt)[method]

        x, t = initial.to(torch.float32), 0.0
        for step in range(steps):
            X[step], T[step] = x, t
            x, t = integrator(x, t)

        # X = X.reshape(*initial.shape, steps).squeeze()
        X = X.squeeze()

        return X, T


if __name__ == "__main__":

    a, b = 0, 4
    T = 1000
    t = torch.linspace(a, b, T)

    legends = []

    plt.figure()
    # plt.xlim([0, 10])
    # plt.ylim([-1, 1])

    # # Real Solution
    # y = (-1 * t).exp()
    # legends += ['true solution']
    # plt.plot(t, y)

    # A = torch.tensor([[-1, 0],
    #                   [0, 0]])
    # initial = torch.tensor([1, 1])

    samples = 16
    # A = torch.tensor([[0, 0, 0, 0],
    #                   [0, 0, 0, 0],
    #                   [4, -2, 0, 0],
    #                   [3, -3, 0, 0]])
    # A = torch.tensor([[4, -2],
    #                   [3, -3]])
    A = torch.tensor([[0.5, -1], [1, -1]])
    initial = (torch.rand(samples, len(A)) - 0.5)
    # initial = torch.tensor([1, 0, 0, 0])
    plt.scatter(initial[:, 0], initial[:, 1])

    # # Euler Integration
    # for dt in [0.8]:
    #     start = time.perf_counter()
    #     y_euler, t = Integrator.integrate(A, initial, dt=dt, steps=T, method='euler')
    #     stop = time.perf_counter()
    #     plt.plot(t, y_euler[:, 0])
    #     legends += [f'euler dt = {dt}']
    #     print(f'{"Euler":10} {dt:5.5f} {stop - start:5.5f}')

    # Runge-Kutta Integration
    for dt in [0.005]:
        start = time.perf_counter()
        x, t = Integrator.integrate(A, initial, dt=dt, steps=T, method='rk4')
        stop = time.perf_counter()
        print(x.shape)
        x = x.squeeze().permute(1, 2, 0)
        for batch in x:
            plt.plot(batch[0], batch[1])
        legends += [f'rk4 dt = {dt}']
        print(f'{"RK4":10} {dt:5.5f} {stop - start:5.5f}')

    plt.legend(legends)
    plt.grid()
    plt.show()
