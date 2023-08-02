import numpy as np
from sys import stdin


def read_cam():
    p = list(map(float, input().split()[1:]))
    return [p[0], p[1]], np.array([[p[2], 0, p[4]], [0, p[3], p[5]], [0, 0, 1]])

def inv(pose): return np.hstack((pose[:3, :3].T, (-pose[:3, :3].T @ pose[:3, 3]).reshape(3,1)))
def mult(p1, p2): return np.hstack((p1[:3, :3] @ p2[:3,:3], (p1[:3,3] + p1[:3,:3] @ p2[:3,3]).reshape(3,1)))

def hat(x): return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
def unhat(x): return np.array([x[2, 1], x[0, 2], x[1, 0]]).reshape(3, 1)


def triang(ray, pose):
    A = np.zeros((6, 4))
    A[:3, :] = hat(ray[:3]) @ pose
    A[3:, :3] = hat(ray[3:])
    x = np.linalg.svd(A)[2][-1]
    return x[:3] / x[3]


def eval_triang(pt, pose, camX, camY):
    y = camY[1] @ pt
    y = y[:2] / y[2]

    pt2 = pose @ np.append(pt, 1)
    x = camX[1] @ pt2
    x = x[:2] / x[2]

    return pt[2] > 0 and pt2[2] > 0 and y[0] > 0 and y[1] > 0 and x[
        0] > 0 and x[1] > 0 and x[0] < camX[0][0] and x[1] < camX[0][1] and y[
            0] < camY[0][0] and y[1] < camY[0][1]


def decompose(E, rays, camX, camY):
    U, E, Vt = np.linalg.svd(E)
    U *= np.linalg.det(U)
    Vt *= np.linalg.det(Vt)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    Z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    t = unhat(U @ Z @ U.T)
    t = t / np.linalg.norm(t)
    poses = [
        np.hstack((U @ W @ Vt, t)),
        np.hstack((U @ W @ Vt, -t)),
        np.hstack((U @ np.linalg.inv(W) @ Vt, t)),
        np.hstack((U @ np.linalg.inv(W) @ Vt, -t))
    ]
    scores = [
        sum([eval_triang(triang(ray, pose), pose, camX, camY) for ray in rays])
        for pose in poses
    ]
    return poses[np.argmax(scores)], np.array([
        np.linalg.norm(triang(ray, poses[np.argmax(scores)])) for ray in rays
    ])


def eight_point(ptsX, ptsY, camX, camY):
    rays = np.zeros((((ptsX[:, 0] > 0) & (ptsY[:, 0] > 0)).sum(), 6))
    A, counter = np.zeros((rays.shape[0], 9)), 0
    for x, y in zip(ptsX, ptsY):
        if x[0] < 0 or y[0] < 0: continue
        x = np.linalg.inv(camX[1]) @ np.append(x, 1)
        y = np.linalg.inv(camY[1]) @ np.append(y, 1)
        for i in range(3):
            A[counter, i * 3:(i + 1) * 3] = x[i] * y
        rays[counter, :3], rays[counter, 3:] = x, y
        counter += 1
    _, _, Vt = np.linalg.svd(A)
    return Vt[-1].reshape(3, 3), rays


cams, pts = [read_cam() for i in range(3)], []
for line in stdin: pts.append(list(map(float, line.split())))

pts = np.array(pts)
E = eight_point(pts[:, :2], pts[:, 2:4], cams[0], cams[1])
p10, d10 = decompose(*E, cams[0], cams[1])
E = eight_point(pts[:, 4:6], pts[:, 2:4], cams[2], cams[1])
p12, d12 = decompose(*E, cams[2], cams[1])

c10 = d10[pts[(pts[:, 1] > 0) & (pts[:, 3] > 0)][:, 5] > 0]
c12 = d12[pts[(pts[:, 5] > 0) & (pts[:, 3] > 0)][:, 1] > 0]
diff = np.mean(c10 / c12)

p10[:,3] /= diff
p = inv(mult(p12, inv(p10)))
p[:3,3] /= np.linalg.norm(p[:3,3])
for i in p: print(*i)
