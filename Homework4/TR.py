import numpy as np


def get_cameras_parameters(cam):
    camera = cam.split()

    camera_parameters = {
        "width": int(camera[1]),
        "height": int(camera[2]),
        "focal_x": int(camera[3]),
        "focal_y": int(camera[4]),
        "center_x": int(camera[5]),
        "center_y": int(camera[6]),
    }

    return camera_parameters


def get_intrinsic_matrix(camera_parameter):
    intrinsic_matrix = np.zeros((3, 3))

    intrinsic_matrix[0, 0] = camera_parameter['focal_x']
    intrinsic_matrix[1, 1] = camera_parameter['focal_y']
    intrinsic_matrix[0, 2] = camera_parameter['center_x']
    intrinsic_matrix[1, 2] = camera_parameter['center_y']
    intrinsic_matrix[2, 2] = 1

    return intrinsic_matrix


def get_landmarks_projections(landmarks, landmarks_projections, idx):
    landmarks_projections[idx] = landmarks.split()

    return landmarks_projections


def get_projections_on_1_and_2_matrix(landmarks_projections):
    projections_on_1_and_2 = np.zeros((32, 4))
    i = 0

    for idx, row in enumerate(landmarks_projections):
        if row[0] != -1 and row[1] != -1:
            projections_on_1_and_2[i] = row[0:4]
            i = i + 1

    projections_on_1_and_2 = projections_on_1_and_2[0:i]
    return projections_on_1_and_2


def get_projections_on_2_and_3_matrix(landmarks_projections):
    projections_on_2_and_3 = np.zeros((32, 4))
    i = 0

    for idx, row in enumerate(landmarks_projections):
        if row[4] != -1 and row[5] != -1:
            projections_on_2_and_3[i] = row[2:6]
            i = i + 1

    projections_on_2_and_3 = projections_on_2_and_3[0:i]
    return projections_on_2_and_3


def get_projections_on_1_and_3_matrix(landmarks_projections):
    projections_on_1_and_3 = np.zeros((32, 6))
    i = 0

    for idx, row in enumerate(landmarks_projections):
        if row[0] != -1 and row[1] != -1 and row[4] != -1 and row[5] != -1:
            projections_on_1_and_3[i] = row
            i = i + 1

    projections_on_1_and_3 = projections_on_1_and_3[0:i]
    return projections_on_1_and_3


def get_normalized_landmarks_projections(inverse_intrinsic_matrix_first, inverse_intrinsic_matrix_second,
                                         landmarks_projections):
    normalized_landmarks_projections = landmarks_projections.copy()
    for idx, row in enumerate(landmarks_projections):
        normalized_landmarks_projections[idx, 0:2] = (inverse_intrinsic_matrix_first @ np.append(row[0:2], [1]))[0:2]
        normalized_landmarks_projections[idx, 2:4] = (inverse_intrinsic_matrix_second @ np.append(row[2:4], [1]))[0:2]

    return normalized_landmarks_projections


def get_q_matrix(landmarks_projections):
    q_matrix = np.zeros((landmarks_projections.shape[0], 9))

    for idx in range(q_matrix.shape[0]):
        q_matrix[idx][0] = landmarks_projections[idx][2] * landmarks_projections[idx][0]
        q_matrix[idx][1] = landmarks_projections[idx][2] * landmarks_projections[idx][1]
        q_matrix[idx][2] = landmarks_projections[idx][2]
        q_matrix[idx][3] = landmarks_projections[idx][3] * landmarks_projections[idx][0]
        q_matrix[idx][4] = landmarks_projections[idx][3] * landmarks_projections[idx][1]
        q_matrix[idx][5] = landmarks_projections[idx][3]
        q_matrix[idx][6] = landmarks_projections[idx][0]
        q_matrix[idx][7] = landmarks_projections[idx][1]
        q_matrix[idx][8] = 1

    return q_matrix


def get_e_matrix(q_matrix):
    U, S, V = np.linalg.svd(q_matrix)
    e = V[-1]
    e_matrix = e.reshape(3, 3)

    return e_matrix


def get_rt_matrix(e_matrix, points, K2):
    U, S, Vt = np.linalg.svd(e_matrix)

    # U，V检查思路：https://stackoverflow.com/questions/22807039/decomposition-of-essential-matrix-validation-of-the-four-possible-solutions-for?rq=3
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2]
    t2 = -U[:, 2]

    T1 = np.hstack((R1, t1.reshape(-1, 1)))
    T2 = np.hstack((R1, t2.reshape(-1, 1)))
    T3 = np.hstack((R2, t1.reshape(-1, 1)))
    T4 = np.hstack((R2, t2.reshape(-1, 1)))

    count = [0, 0, 0, 0]  # 计数z>0的R和t组合出现次数

    for idx in range(points.shape[0]):
        p_2_x = points[idx, 2]
        p_2_y = points[idx, 3]
        p_2_z = 1

        p_2_skew_symmetric_matrix = np.array([[0, -p_2_z, p_2_y],
                                              [p_2_z, 0, -p_2_x],
                                              [-p_2_y, p_2_x, 0]])
        M2 = [K2 @ T1, K2 @ T2, K2 @ T3, K2 @ T4]

        for i in range(4):  # 四种T
            A = p_2_skew_symmetric_matrix @ M2[i]
            _, _, V_p_2_t = np.linalg.svd(A)
            P_world = V_p_2_t[-1]  # 3D P坐标在第一个坐标系下
            P_world = P_world / P_world[-1]

            P_world_2 = M2[i] @ P_world  # 3D P坐标在第二个坐标系下 使用Rt矩阵转移

            # 计数检查思路:https://www.cnblogs.com/houkai/p/6665506.html
            if P_world[2] > 0 and P_world_2[2] > 0:  # 遍历所有的landmarks， 计数使用的Rt矩阵组合
                count[i] = count[i] + 1

    output = [count[0] + count[1], count[2] + count[3]]  # 1和2都是使用R1+不同的t；3和4都是使用R2+不同的t

    if output[0] > output[1]:
        if count[0] > count[1]:
            return T1
        elif count[0] < count[1]:
            return T2
        else:
            raise EOFError('选取R,t异常')
    elif output[0] < output[1]:
        if count[2] > count[3]:
            return T3
        elif count[2] < count[3]:
            return T4
        else:
            raise EOFError('选取R,t异常')
    else:
        raise EOFError('选取R,t异常')


def get_transpose_T_matrix(T):
    R = T[0:3, 0:3]
    R_I = np.linalg.inv(R)

    t = T[0:3, 3]
    t_I = -R_I @ t

    T = np.hstack((R_I, t_I.reshape(-1, 1)))
    T = np.vstack((T, [0, 0, 0, 1]))

    return T


def get_scaled_T_matrix(T1, T2, K1, K2, K3, points):
    # 详见2d_2d_slide P10
    T1_I = get_transpose_T_matrix(T1)
    R1 = T1[0:3, 0:3]
    t1 = T1[0:3, 3]

    R2 = T2[0:3, 0:3]
    t2 = T2[0:3, 3]

    u1, v1, u2, v2, u3, v3 = points[0]

    # 将pixel coordinate 从向量p转为负对称矩阵形式p_x
    p_1_x = np.array([
        [0., -1., v1],
        [1., 0., -u1],
        [-v1, u1, 0.]
    ])

    p_2_x = np.array([
        [0., -1., v2],
        [1., 0., -u2],
        [-v2, u2, 0.]
    ])

    p_3_x = np.array([
        [0., -1., v3],
        [1., 0., -u3],
        [-v3, u3, 0.]
    ])

    I_0 = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    # 构建M矩阵， M=KRT
    # 此处的目的是将上述点在第二个坐标系下表示
    M1 = K1 @ T1_I[0:3]
    M2 = K2 @ I_0
    M3 = K3 @ T2[0:3]

    # 这里把两个Q拼接起来做SVD，求解P，不清楚原因
    Q1 = np.vstack((p_1_x @ M1, p_2_x @ M2))
    Q2 = np.vstack((p_2_x @ M2, p_3_x @ M3))

    _, _, v_1_t = np.linalg.svd(Q1)
    _, _, v_2_t = np.linalg.svd(Q2)

    v_1 = v_1_t[-1]
    v_2 = v_2_t[-1]

    v_1 = v_1 / v_1[-1]  # X, Y, Z, 1
    v_2 = v_2 / v_2[-1]  # X, Y, Z, 1

    # 基于第二个坐标系表示的一点/三点之间存在一个比例关系
    ratios = [v1 / v2 for v1, v2 in zip(v_1[0:2], v_2[0:2])]
    sum_ratios = sum(ratios)
    mean_ratio = sum_ratios / len(ratios)

    t2 = mean_ratio * t2
    T1_new = np.hstack((R1, t1.reshape(-1, 1)))
    T1_new = np.vstack((T1_new, [0, 0, 0, 1]))
    T2_new = np.hstack((R2, t2.reshape(-1, 1)))
    T2_new = np.vstack((T2_new, [0, 0, 0, 1]))

    T13 = T2_new @ T1_new
    T31 = get_transpose_T_matrix(T13)
    R31 = T31[0:3, 0:3]
    t31 = T31[0:3, 3]
    t31_norm = np.linalg.norm(t31)
    t31 = t31 / t31_norm

    T31_new = np.hstack((R31, t31.reshape(-1, 1)))

    return T31_new


def print_outputs(T):
    output = ''

    for i in range(3):
        for j in range(4):
            output = output + "%.6f" % T[i][j]
            if j != 3:
                output = output + " "
        output = output + "\n"

    print(output)

    return None


if __name__ == '__main__':
    # camera data
    cameras_parameters = []
    for camera_id in range(3):
        camera_parameters_input = input()
        cameras_parameters.append(get_cameras_parameters(camera_parameters_input))  # 3

    # matrix K and K.I
    for idx in range(3):
        if idx == 0:
            intrinsic_matrix_1 = get_intrinsic_matrix(cameras_parameters[idx])
        elif idx == 1:
            intrinsic_matrix_2 = get_intrinsic_matrix(cameras_parameters[idx])
        elif idx == 2:
            intrinsic_matrix_3 = get_intrinsic_matrix(cameras_parameters[idx])

    intrinsic_matrix_1_I = np.linalg.inv(intrinsic_matrix_1)
    intrinsic_matrix_2_I = np.linalg.inv(intrinsic_matrix_2)
    intrinsic_matrix_3_I = np.linalg.inv(intrinsic_matrix_3)

    # landmarks projections
    landmarks_projections = np.zeros((32, 6))
    idx = 0
    while True:
        try:
            landmarks_projections_input = input()
            landmarks_projections = get_landmarks_projections(landmarks_projections_input, landmarks_projections, idx)
            idx = idx + 1
        except:
            break
    landmarks_projections = landmarks_projections[0:idx]

    # landmarks projections between cameras
    projections_on_1_and_2 = get_projections_on_1_and_2_matrix(landmarks_projections)
    projections_on_2_and_3 = get_projections_on_2_and_3_matrix(landmarks_projections)
    projections_on_1_and_3 = get_projections_on_1_and_3_matrix(landmarks_projections)

    # Normalize landmarks projections
    normalized_projections_on_1_and_2 = get_normalized_landmarks_projections(intrinsic_matrix_1_I, intrinsic_matrix_2_I,
                                                                             projections_on_1_and_2)
    normalized_projections_on_2_and_3 = get_normalized_landmarks_projections(intrinsic_matrix_2_I, intrinsic_matrix_3_I,
                                                                             projections_on_2_and_3)

    # Q matrix
    q_matrix_1_and_2 = get_q_matrix(normalized_projections_on_1_and_2)
    q_matrix_2_and_3 = get_q_matrix(normalized_projections_on_2_and_3)

    # E matrix
    e_matrix_1_and_2 = get_e_matrix(q_matrix_1_and_2)
    e_matrix_2_and_3 = get_e_matrix(q_matrix_2_and_3)

    # R,t matrix
    T1 = get_rt_matrix(e_matrix_1_and_2, projections_on_1_and_2, intrinsic_matrix_2)
    T2 = get_rt_matrix(e_matrix_2_and_3, projections_on_2_and_3, intrinsic_matrix_3)

    T1 = np.vstack((T1, [0, 0, 0, 1]))
    T2 = np.vstack((T2, [0, 0, 0, 1]))

    # scaled T matrix
    T31 = get_scaled_T_matrix(T1, T2, intrinsic_matrix_1, intrinsic_matrix_2, intrinsic_matrix_3,
                              projections_on_1_and_3)

    # print output
    print_outputs(T31)
