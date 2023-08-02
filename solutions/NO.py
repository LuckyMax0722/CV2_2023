"""
In this assignment we just need to implement formulas from the link in the problem description

SO3, SE3 exp maps can just be copied from internet, implemented by ChatGPT or as one of the students did: 

they can be implemented through Tailor series:
def expm(A):
    sol = np.zeros_like(A, dtype=np.float64)
    for k in range(100):
        A_k = np.linalg.matrix_power(A, k)
        sol += 1.0 / math.factorial(k) * A_k
    return sol


Jacobians and Gauss-Newton steps are given by explicit formulas
"""
import numpy as np

def read(): s = input(); return s if len(s) > 0 else read()
pts_3d = np.array([np.array([i,j,k]) for i in [0,1] for j in [0,1] for k in[0,1]])
pts_2d = np.array([np.array(input().split()[:2]).astype(float) for i in range(8)])

K, T = np.zeros((3, 3)), np.eye(4)
for i in range(3): K[i,:] = np.array(read().split(), dtype=float)
for i in range(3): T[i,:] = np.array(read().split(), dtype=float)

def so3_exponent_map(a):
    theta_norm = np.linalg.norm(a)
    if theta_norm < 1e-6: return np.eye(3)
    theta_hat = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])

    return np.eye(3) + np.sin(theta_norm) / theta_norm * theta_hat + \
        (1 - np.cos(theta_norm)) / (theta_norm**2) * theta_hat @ theta_hat

def se3_exponent_map(xi):
    v = np.array([xi[0], xi[1], xi[2]])
    omega = np.array([xi[3], xi[4], xi[5]])
    
    theta = np.linalg.norm(omega)

    if theta < 1e-16: return np.vstack((np.hstack((np.eye(3), v[:, np.newaxis])), np.array([0, 0, 0, 1])))
    
    omega_hat = np.array([[0, -omega[2], omega[1]], [omega[2], 0, -omega[0]], [-omega[1], omega[0], 0]]) / theta
    
    R = so3_exponent_map(omega)
    G = np.eye(3) + ((1 - np.cos(theta)) / theta**2) * omega_hat + \
        ((theta - np.sin(theta)) / theta**3) * omega_hat @ omega_hat
    
    return np.vstack((np.hstack((R, (G @ v)[:, np.newaxis])), np.array([0, 0, 0, 1])))

def residual():
    pts = (pts_3d @ T[:3,:3].T + T[:3,3].T) @ K.T
    pts = pts[...,:2] / pts[...,2].reshape(-1, 1)
    return pts - pts_2d

def jac(): 
    O = np.zeros(pts_3d.shape[0])
    I = np.ones(pts_3d.shape[0])
    g = pts_3d @ T[:3, :3].T + T[:3, 3].T
    gx, gy, gz = g[...,0], g[...,1], g[...,2]
    T_jac = np.stack([K[0, 0] / gz, O, -K[0, 0] * gx / gz**2, -K[0, 0] * gx * gy / gz**2, K[0, 0] * (1 + (gx /gz)**2), -K[0, 0] * gy / gz,
                      O, K[1, 1] / gz, -K[1, 1] * gy / gz**2, -K[1, 1] * (1 + (gy /gz)**2), K[1, 1] * gx * gy / gz**2, K[1, 1] * gx / gz], axis=-1).reshape(-1, 2, 6)
    K_jac = np.stack([gx/gz, O, I, O, O, gy/gz, O, I], axis=-1).reshape(-1, 2, 4)

    return np.concatenate([K_jac, T_jac], axis=-1)

for i in range(10):
    res = residual()
    J = jac()
    b = np.mean(np.einsum('Bij,Bi->Bj',  J, res), axis=0)
    H = np.mean(np.einsum('Bij,Bik->Bjk', J, J), axis=0)
    res = np.linalg.solve(H, -b)
    T_new = se3_exponent_map(res[4:])
    T = T_new @ T
    K[0, 0] += res[0]; K[1, 1] += res[1]; K[0, 2] += res[2]; K[1, 2] += res[3]

    for j in range(3): print(*K[j])
    print()
    for j in range(3): print(*T[j])
    print()
    print()
