import numpy as np
import math


def get_camera_parameters(camera_paras, cam_paras):
    camera = camera_paras.split()

    if camera[0] == "pinhole":
        cam_paras = {
            "width": int(camera[1]),
            "height": int(camera[2]),
            "focal_x": int(camera[3]),
            "focal_y": int(camera[4]),
            "center_x": int(camera[5]),
            "center_y": int(camera[6]),
            "w": int(-5000)
        }
    else:
        cam_paras = {
            "width": int(camera[1]),
            "height": int(camera[2]),
            "focal_x": int(camera[3]),
            "focal_y": int(camera[4]),
            "center_x": int(camera[5]),
            "center_y": int(camera[6]),
            "w": float(camera[7])
        }

    return cam_paras


def get_calibration_matrix(cam_para):
    calibration_matrix = np.array([[cam_para["focal_x"], 0, cam_para["center_x"]],
                                   [0, cam_para["focal_y"], cam_para["center_y"]],
                                   [0, 0, 1]])
    return calibration_matrix


def print_outputs(i, output, out, cam_boundary_x, cam_boundary_y):
    if cam_boundary_x >= out[0] >= 0 and cam_boundary_y >= out[1] >= 0:
        if i != 8:
            output = output + "%.3f" % out[0]
            output = output + " "
            output = output + "%.3f" % out[1]
            output = output + "\n"
        else:
            output = output + "%.3f" % out[0]
            output = output + " "
            output = output + "%.3f" % out[1]
    else:
        if i != 8:
            output = output + "OB"
            output = output + "\n"
        else:
            output = output + "OB"

    return output


# camera data
camera_1_input = input()
cam_1 = {}
cam_1 = get_camera_parameters(camera_1_input, cam_1)
camera_2_input = input()
cam_2 = {}
cam_2 = get_camera_parameters(camera_2_input, cam_2)

# RT matrix
rt_matrix = np.zeros((3, 4))
for i in range(3):
    rt_matrix[i] = input().split()

# K/Calibration matrix
k1 = get_calibration_matrix(cam_1)
k2 = get_calibration_matrix(cam_2)

#
points_matrix = np.empty((9, 3))
i = 0
while True:
    try:
        point = input().split()
        for j in range(3):
            points_matrix[i][j] = point[j]
        i += 1
    except:
        break

output = ""
cam_boundary_x = 2 * cam_2["center_x"]
cam_boundary_y = 2 * cam_2["center_y"]
z_negative = False


if cam_1["w"] == -5000 and cam_2["w"] != -5000:
    for i in range(points_matrix.shape[0]):
        x_c = np.linalg.inv(k1) @ np.append(points_matrix[i][0:2], [1]).T
        z_squared = points_matrix[i][2] ** 2 / (x_c[0] ** 2 + x_c[1] ** 2 + 1)
        z = math.sqrt(z_squared)
        x_c = x_c * z
        x_c = np.append(x_c, [1])
        out = rt_matrix @ x_c.T

        if out[2] <= 0:
            z_negative = True

        out = out / out[2]

        r = math.sqrt(out[0] ** 2 + out[1] ** 2)
        g = (1 / (cam_2["w"] * r)) * np.arctan(2 * r * math.tan(cam_2["w"] / 2))

        out[0:2] = out[0:2] * g

        out = k2 @ out

        if z_negative:
            out = out * 10000

        z_negative = False

        output = print_outputs(i, output, out, cam_boundary_x, cam_boundary_y)

elif cam_1["w"] != -5000 and cam_2["w"] == -5000:
    for i in range(points_matrix.shape[0]):
        x_c = np.linalg.inv(k1) @ np.append(points_matrix[i][0:2], [1]).T

        dr = np.linalg.norm(x_c[0:2])

        r = np.tan(dr * cam_1["w"]) / (2 * np.tan(cam_1["w"] / 2))
        x_c_d = r * x_c[0:2] / dr

        z_squared = points_matrix[i][2] ** 2 / (x_c_d[0] ** 2 + x_c_d[1] ** 2 + 1)
        z = math.sqrt(z_squared)
        x_c = [x_c_d[0] * z, x_c_d[1] * z, z]

        x_c = np.append(x_c, [1])

        out =  rt_matrix @ x_c.T

        if out[2] <= 0:
            z_negative = True

        out = k2 @ out
        out = out / out[2]

        if z_negative:
            out = out * 10000

        z_negative = False

        output = print_outputs(i, output, out, cam_boundary_x, cam_boundary_y)

elif cam_1["w"] != -5000 and cam_2["w"] != -5000:
    for i in range(points_matrix.shape[0]):
        x_c = np.linalg.inv(k1) @ np.append(points_matrix[i][0:2], [1]).T

        dr = np.linalg.norm(x_c[0:2])

        r = np.tan(dr * cam_1["w"]) / (2 * np.tan(cam_1["w"] / 2))
        x_c_d = r * x_c[0:2] / dr

        z_squared = points_matrix[i][2] ** 2 / (x_c_d[0] ** 2 + x_c_d[1] ** 2 + 1)
        z = math.sqrt(z_squared)
        x_c = [x_c_d[0] * z, x_c_d[1] * z, z]

        x_c = np.append(x_c, [1])

        out = rt_matrix @ x_c.T

        if out[2] <= 0:
            z_negative = True

        out = out / out[2]

        r_n = math.sqrt(out[0] ** 2 + out[1] ** 2)

        g = (1 / (cam_2["w"] * r_n)) * np.arctan(2 * r_n * math.tan(cam_2["w"] / 2))


        out[0:2] = out[0:2] * g

        out = k2 @ out

        if z_negative:
            out = out * 10000

        z_negative = False

        output = print_outputs(i, output, out, cam_boundary_x, cam_boundary_y)

elif cam_1["w"] == -5000 and cam_2["w"] == -5000:
    for i in range(points_matrix.shape[0]):
        x_c = np.linalg.inv(k1) @ np.append(points_matrix[i][0:2], [1]).T
        z_squared = points_matrix[i][2] ** 2 / (x_c[0] ** 2 + x_c[1] ** 2 + 1)
        z = math.sqrt(z_squared)
        x_c = x_c * z
        x_c = np.append(x_c, [1])

        out = rt_matrix @ x_c.T
        if out[2] <= 0:
            z_negative = True

        out = k2 @ out
        out = out / out[2]

        if z_negative:
            out = out * 10000

        z_negative = False

        output = print_outputs(i, output, out, cam_boundary_x, cam_boundary_y)

print(output)
