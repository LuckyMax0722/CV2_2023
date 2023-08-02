import numpy as np
from math import tan, atan
from sys import stdin


class Camera:

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def oob(self, pt):
        return pt[0] >= 0 and pt[0] < (self.w-1) and pt[1] >= 0 and pt[1] < (self.h-1)

    def project(self, pt):
        if pt[2] < 0: return None, False
        pt = self._project(pt)
        return pt, self.oob(pt)

    def unproject(self, pt, d):
        pt, valid = self._unproject(pt)
        return d * pt / np.linalg.norm(pt), valid


class Pinhole(Camera):

    def __init__(self, w, h, fx, fy, cx, cy):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def _project(self, pt):
        pt = self.K @ pt
        return pt[:2] / pt[2]

    def _unproject(self, pt):
        return (np.linalg.inv(self.K) @ np.append(pt, 1)), True


class Fov(Camera):

    def __init__(self, w, h, fx, fy, cx, cy, W):
        Camera.__init__(self, w, h)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.W = W

    def _project(self, pt):
        pt = pt[:2] / pt[2]
        r = np.linalg.norm(pt)
        r_new = 1.0 / self.W * atan(2 * r * tan(self.W / 2))
        return (self.K @ np.append((r_new / r) * pt, 1))[:2]

    def _unproject(self, pt):
        pt = (np.linalg.inv(self.K) @ np.append(pt, 1))[:2]
        r = np.linalg.norm(pt)
        r_new = tan(r * self.W) / 2 / tan(self.W / 2)
        return np.append((r_new / r) * pt, 1), r_new > 0


def read_camera():
    m = input().split()
    m_n = m[0]
    if m_n == "pinhole": return Pinhole(*list(map(float, m[1:])))
    if m_n == "fov": return Fov(*list(map(float, m[1:])))


def T(pt, pose):
    return pose @ np.append(pt, 1)


m_i = read_camera()
m_o = read_camera()
pose = np.zeros((3, 4))
for i in range(3):
    pose[i, :] = np.array(list(map(float, input().split())))

for line in stdin:
    line = list(map(float, line.split()))
    pt = np.array(line[:2])
    d = line[2]
    pt, v  = m_i.unproject(pt, d)
    if v:
        pt = T(pt, pose)
        pt, v = m_o.project(pt)
    if not v: print("OB")
    else: print(f'{pt[0]} {pt[1]}')
