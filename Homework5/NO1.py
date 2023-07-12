import numpy as np


def get_unit_cube_projections():
    vertices_projections = np.zeros((8, 2))
    for i in range(8):
        vertices_projections[i] = input().split()

    return vertices_projections


def get_k_matrix():
    input()  # 跳过空行
    k = np.zeros((3, 3))
    for i in range(3):
        k[i] = input().split()

    return k


def get_rt_matrix():
    input()  # 跳过空行
    rt = np.zeros((3, 4))
    for i in range(3):
        rt[i] = input().split()

    return rt


def get_residuals(T, vertices_projections, vertices_world_frame):
    residuals = []
    scale = []
    x_cam_unscaled = []

    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam = T @ vertice_world_frame_argumented  # KRT 映射到相机平面上
        x_cam_unscaled.append(x_cam)
        x_cam_z = x_cam[2]
        scale.append(x_cam_z)
        x_cam = x_cam / x_cam_z  # Z归一

        # 计算residuals
        residuals_1 = x_cam[0] - vertices_projections[i, 0]
        residuals_2 = x_cam[1] - vertices_projections[i, 1]
        residuals.append(residuals_1)
        residuals.append(residuals_2)

    return residuals, scale, x_cam_unscaled


def get_dfx(rt, scale, vertices_world_frame, J):
    k = np.zeros((3, 3))
    k[0, 0] = 1  # dfx
    T = k @ rt

    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam = T @ vertice_world_frame_argumented  # KRT 映射到相机平面上
        x_cam = x_cam / scale[i]  # 这里要up to scale
        J[2 * i, 0] = x_cam[0]
        J[2 * i + 1, 0] = x_cam[1]

    return J


def get_dfy(rt, scale, vertices_world_frame, J):
    k = np.zeros((3, 3))
    k[1, 1] = 1  # dfx
    T = k @ rt

    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam = T @ vertice_world_frame_argumented  # KRT 映射到相机平面上
        x_cam = x_cam / scale[i]  # 这里要up to scale
        J[2 * i, 1] = x_cam[0]
        J[2 * i + 1, 1] = x_cam[1]

    return J


def get_dcx(rt, scale, vertices_world_frame, J):
    k = np.zeros((3, 3))
    k[0, 2] = 1  # dfx
    T = k @ rt

    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam = T @ vertice_world_frame_argumented  # KRT 映射到相机平面上
        x_cam = x_cam / scale[i]  # 这里要up to scale
        J[2 * i, 2] = x_cam[0]
        J[2 * i + 1, 2] = x_cam[1]

    return J


def get_dcy(rt, scale, vertices_world_frame, J):
    k = np.zeros((3, 3))
    k[1, 2] = 1  # dfx
    T = k @ rt

    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam = T @ vertice_world_frame_argumented  # KRT 映射到相机平面上
        x_cam = x_cam / scale[i]  # 这里要up to scale
        J[2 * i, 3] = x_cam[0]
        J[2 * i + 1, 3] = x_cam[1]

    return J


def get_jacobian(k, rt, vertices_world_frame, J):
    for i in range(8):
        vertice_world_frame_argumented = np.append(vertices_world_frame[i], 1)  # 补1
        x_cam_unscaled = rt @ vertice_world_frame_argumented  # RT 映射到相机平面上
        J[2 * i, 4] = k[0, 0] / x_cam_unscaled[2]
        J[2 * i, 6] = -1 * k[0, 0] * x_cam_unscaled[0] / x_cam_unscaled[2] ** 2
        J[2 * i, 7] = -1 * k[0, 0] * x_cam_unscaled[0] * x_cam_unscaled[1] / x_cam_unscaled[2] ** 2
        J[2 * i, 8] = k[0, 0] * (1 + x_cam_unscaled[0] ** 2 / x_cam_unscaled[2] ** 2)
        J[2 * i, 9] = -1 * k[0, 0] * x_cam_unscaled[1] / x_cam_unscaled[2]

        J[2 * i + 1, 5] = k[1, 1] / x_cam_unscaled[2]
        J[2 * i + 1, 6] = -1 * k[1, 1] * x_cam_unscaled[1] / x_cam_unscaled[2] ** 2
        J[2 * i + 1, 7] = -1 * k[1, 1] * (1 + x_cam_unscaled[1] ** 2 / x_cam_unscaled[2] ** 2)
        J[2 * i + 1, 8] = k[1, 1] * x_cam_unscaled[0] * x_cam_unscaled[1] / x_cam_unscaled[2] ** 2
        J[2 * i + 1, 9] = k[1, 1] * x_cam_unscaled[0] / x_cam_unscaled[2]

    return J


def print_outputs(residuals, J):
    output = ''

    for i in range(16):
        if i == 15:
            output = output + "%.16f" % residuals[i] + "\n"
        else:
            output = output + "%.16f" % residuals[i] + " "

    output = output + "\n"

    for i in range(16):
        for j in range(10):
            if j == 9:
                output = output + "%.16f" % J[i][j] + "\n"
            else:
                output = output + "%.16f" % J[i][j] + " "
    print(output)

    return None


if __name__ == '__main__':
    #  获取input数据
    vertices_projections = get_unit_cube_projections()
    k = get_k_matrix()
    rt = get_rt_matrix()
    vertices_world_frame = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

    #  get_residuals, T = KRT
    T = k @ rt
    residuals, scale, x_cam_unscaled = get_residuals(T, vertices_projections, vertices_world_frame)

    J = np.zeros((16, 10), dtype=float)
    J = get_dfx(rt, scale, vertices_world_frame, J)  # dfx
    J = get_dfy(rt, scale, vertices_world_frame, J)  # dfy
    J = get_dcx(rt, scale, vertices_world_frame, J)  # dcx
    J = get_dcy(rt, scale, vertices_world_frame, J)  # dcy
    J = get_jacobian(k, rt, vertices_world_frame, J)  # jacobian

    print_outputs(residuals, J)
