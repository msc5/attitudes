
import numpy as np


class vector:

    def cross(v):
        v1, v2, v3 = v.squeeze()
        return np.array([
            [0, -v3, v2],
            [v3, 0, -v1],
            [-v2, v1, 0]
        ])


class quaternion:

    def from_theta(e, theta):
        n = 1 if not hasattr(theta, '__len__') else len(theta)
        q = np.append(e.reshape((3, 1)) * np.sin(theta / 2), np.cos(theta / 2))
        return q.reshape((4, n))

    def psi(q):
        qv = q[0:3]
        q4 = q[3]
        qcross = vector.cross(qv)
        a = q[3] * np.eye(3) - qcross
        b = - (qv.T).reshape((1, 3))
        return np.concatenate((a, b), axis=0)

    def xi(q):
        qv = q[0:3]
        q4 = q[3]
        qcross = vector.cross(qv)
        a = q[3] * np.eye(3) + qcross
        b = - (qv.T).reshape((1, 3))
        return np.concatenate((a, b), axis=0)

    def cross(q):
        return np.append(quaternion.psi(q), q, axis=1)

    def dot(q):
        return np.append(quaternion.xi(q), q, axis=1)

    def A(q):
        return quaternion.xi(q).T @ quaternion.psi(q)


if __name__ == "__main__":

    from rich import print

    q = quaternion

    rot = q.from_theta(np.array([0, 0, 1]), np.pi / 2)
    print(rot, rot.shape)

    rot_xi = q.xi(rot)
    print(rot_xi, rot_xi.shape)

    rot_cross = q.cross(rot)
    print(rot_cross, rot_cross.shape)

    rot_A = q.A(rot)
    print(rot_A, rot_A.shape)
