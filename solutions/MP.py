"""
We need to find a common point for two affine subspaces. 
These subspaces are given by sets of points that span both of them.

To simplify the problem and treat it like a linear space, we first need to shift the subspaces to the origin (zero). 
We can do this by subtracting any point lying inside either subspace from all the spanning points of both subspaces.

Once we perform this shift, we end up with two sets of vectors that span their respective linera subspaces. 
We can then construct the following system of equations:

A   -- Matrix, where the rows are from the first affine subspace, A[i] - row `i`
B   -- Matrix, where the rows are from the second affine subspace, B[i] - row `i`

(A - A[0])^T x + A[0] = (B - B[0])^T y + B[0]

We solve for `x` and `y`

`(A - A[0])^T` columnwise can be seen as an overdetermined basis of a linear space, which should be shifted by A[0]

                               x
[(A - A[0])^T, (-B + B[0])^T]  y  = B[0] - A[0]

And the final answer is `(A - A[0])^T x + A[0]`
"""
import numpy as np

pts_A, pts_B = [], []

m = int(input())
for pts in [pts_A, pts_B]:
    for i in range(int(input())):
        pts.append(list(map(float, input().split()[1:])))
pts_A, pts_B = np.array(pts_A), np.array(pts_B)

b1, b2 = pts_A[0].copy(), pts_B[0].copy()
A = np.hstack(((pts_A - b1).T, (-pts_B + b2).T))

ans = np.linalg.lstsq(A, b2 - b1, rcond=-1)[0]

if np.linalg.norm(A @ ans - b2 + b1) > 1e-4: print("N")
else:
    ans = A[:, :pts_A.shape[0]] @ ans[:pts_A.shape[0]] + b1
    print("Y")
    print(*ans)
