import numpy as np
pts_3d = [np.array([i,j,k,1]) for i in [0,1] for j in [0,1] for k in[0,1]]
pts_2d = [np.array(input().split()[:2]).astype(float) for i in range(8)]

Q = np.zeros((2 * len(pts_3d), 12))
for i in range(len(pts_3d)):
    Q[2 * i, :4] = pts_3d[i]
    Q[2 * i + 1, 4:8] = pts_3d[i]

    Q[2 * i, 8:] = -pts_2d[i][0] * pts_3d[i]
    Q[2 * i + 1, 8:] = -pts_2d[i][1] * pts_3d[i]

_, _, Vt = np.linalg.svd(Q)
m = Vt.T[:, -1].reshape(3, 4)
m /= np.linalg.norm(m[2,:3])

def RQ(A):
    P = np.eye(A.shape[0])
    P = P[:, ::-1]
    A = P @ A
    Q, R = np.linalg.qr(A.T)
    return P @ R.T @ P, P @ Q.T

R, Q = RQ(m[:3, :3])
sign_R = np.eye(3)
for i in range(3): 
    if R[i,i] < 0: sign_R[i, i] = -1
R = R @ sign_R
Q = sign_R @ Q

if np.linalg.det(Q) < 0: 
    Q *= (-1)
    m[:3] *= (-1)
R[0,1] = 0

tran = np.zeros(3)
tran[2] = m[2,3]
tran[0] = (m[0,3] - R[0,2] * tran[2]) / R[0,0]
tran[1] = (m[1,3] - R[1,2] * tran[2]) / R[1,1]

my_K = R / R[2,2]
my_M = np.hstack((Q, tran.reshape(3,1)))
for i in range(3): print(*my_K[i,:])
print()
for i in range(3): print(*my_M[i,:])
