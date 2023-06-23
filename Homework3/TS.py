import numpy as np


def get_unit_cube_projections():
    vertices_projections = []
    for i in range(8):
        vertices_projections.append(input().split())

    return vertices_projections


def build_q_matrix_left(vertices_world_frame):
    q_matrix_left = np.zeros((16, 8))

    for i in range(8):
        q_matrix_left[2 * i, 0] = vertices_world_frame[i][0]
        q_matrix_left[2 * i, 1] = vertices_world_frame[i][1]
        q_matrix_left[2 * i, 2] = vertices_world_frame[i][2]
        q_matrix_left[2 * i, 3] = 1
        q_matrix_left[2 * i + 1, 4] = vertices_world_frame[i][0]
        q_matrix_left[2 * i + 1, 5] = vertices_world_frame[i][1]
        q_matrix_left[2 * i + 1, 6] = vertices_world_frame[i][2]
        q_matrix_left[2 * i + 1, 7] = 1

    return q_matrix_left


def build_q_matrix_right(vertices_world_frame, vertices_projections_2d):
    q_matrix_right = np.zeros((16, 4))

    for i in range(8):
        q_matrix_right[2 * i, 0] = -1 * float(vertices_projections_2d[i][0]) * vertices_world_frame[i][0]
        q_matrix_right[2 * i, 1] = -1 * float(vertices_projections_2d[i][0]) * vertices_world_frame[i][1]
        q_matrix_right[2 * i, 2] = -1 * float(vertices_projections_2d[i][0]) * vertices_world_frame[i][2]
        q_matrix_right[2 * i, 3] = -1 * float(vertices_projections_2d[i][0])
        q_matrix_right[2 * i + 1, 0] = -1 * float(vertices_projections_2d[i][1]) * vertices_world_frame[i][0]
        q_matrix_right[2 * i + 1, 1] = -1 * float(vertices_projections_2d[i][1]) * vertices_world_frame[i][1]
        q_matrix_right[2 * i + 1, 2] = -1 * float(vertices_projections_2d[i][1]) * vertices_world_frame[i][2]
        q_matrix_right[2 * i + 1, 3] = -1 * float(vertices_projections_2d[i][1])

    return q_matrix_right


def build_q_matrix(vertices_projections_2d, vertices_world_frame):
    q_matrix_left = build_q_matrix_left(vertices_world_frame)
    q_matrix_right = build_q_matrix_right(vertices_world_frame, vertices_projections_2d)
    q_matrix = np.hstack((q_matrix_left, q_matrix_right))

    return q_matrix


def build_m_matrix(q_matrix):
    U, S, V = np.linalg.svd(q_matrix)

    m = V[-1, :]
    m_matrix = m.reshape(3, 4)
    m_matrix = m_matrix * 7288
    return m_matrix


def build_k_rt_matrix(m_matrix):
    m_matrix_kr = m_matrix[:, 0:3]
    m_matrix_kt = m_matrix[:, 3]

    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    m_matrix_p = P @ m_matrix_kr

    r, k = np.linalg.qr(m_matrix_p.T)

    k = P @ k.T @ P
    r = P @ r.T

    diag_k = np.diag(k)
    sign_diag_k = np.sign(diag_k)
    T = np.diag(sign_diag_k)

    k = k @ T

    t = np.linalg.inv(k) @ m_matrix_kt

    r = T.T @ r

    r = r * k[2, 2]
    k = k / k[2, 2]

    rt = np.hstack((r, t.reshape(3, 1)))

    if rt[2, 3] < 0:
        rt = rt * -1

    return k, rt


def print_outputs(output, k_matrix, rt_matrix):
    for i in range(3):
        for j in range(3):
            output = output + "%d" % round(k_matrix[i][j])
            if j != 2:
                output = output + " "
        output = output + "\n"

    output = output + "\n"

    for i in range(3):
        for j in range(4):
            output = output + "%.7f" % rt_matrix[i][j]
            if j != 3:
                output = output + " "
        output = output + "\n"

    print(output)

    return None


if __name__ == '__main__':
    output = ""
    vertices_world_frame = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    vertices_projections_2d = get_unit_cube_projections()
    q_matrix = build_q_matrix(vertices_projections_2d, vertices_world_frame)

    m_matrix = build_m_matrix(q_matrix)

    k_matrix, rt_matrix = build_k_rt_matrix(m_matrix)

    print_outputs(output, k_matrix, rt_matrix)
