
import numpy as np


class quaternion:

    def from_theta(e, theta):
        n = len(theta)
        q = np.append(e.reshape((3, 1)) * np.sin(theta / 2), np.cos(theta / 2))
        return q.reshape((4, n))

    def psi(q):
        qv = q[0:3]
        q4 = q[3]
        qcross = np.cross(qv.squeeze(), np.eye(3) - 1)
        return np.append(q[3] * np.eye(3) - qcross, - qv.T, axis=0)

    def xi(q):
        qv = q[0:3]
        q4 = q[3]
        qcross = np.cross(qv.squeeze(), np.eye(3) - 1)
        return np.append(q[3] * np.eye(3) + qcross, - qv.T, axis=0)

    def cross(q):
        return np.append(quaternion.psi(q), q, axis=1)

    def dot(q):
        return np.append(quaternion.xi(q), q, axis=1)

    def A(q):
        qv = q[0:3]
        q4 = q[3]
        qcross = np.cross(qv.squeeze(), np.eye(3) - 1)
        return ((q4**2 - np.linalg.norm(qv)**2) * np.eye(3) -
                2 * q4 * qcross + 2 * qv * qv.T)


if __name__ == "__main__":

    from rich import print

    q = quaternion

    rot = q.from_theta(np.array([0, 0, 1]), np.pi / 4)
    print(rot, rot.shape)

    rot_xi = q.xi(rot)
    print(rot_xi, rot_xi.shape)

    rot_cross = q.cross(rot)
    print(rot_cross, rot_cross.shape)

    rot_A = q.A(rot)
    print(rot_A, rot_A.shape)
