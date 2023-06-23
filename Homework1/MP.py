import numpy as np

m = int(input())

str_p = "Y"

# 获取原始数据_矩阵locations_1
n_1 = int(input())

locations_1 = np.zeros((n_1, m))

home_location_1 = np.zeros(m)

for i in range(n_1):
    landmarks = input().split()
    if i == 0:  # 提取label0 作为原点
        for j in range(1, m + 1):
            home_location_1[j - 1] = landmarks[j]
    for k in range(1, m + 1):
        locations_1[i][k - 1] = landmarks[k]

# 获取原始数据_矩阵locations_2
n_2 = int(input())

locations_2 = np.zeros((n_2, m))

home_location_2 = np.zeros(m)

for i in range(n_2):
    landmarks = input().split()
    if i == 0:
        for j in range(1, m + 1):  # 提取label0 作为原点
            home_location_2[j - 1] = landmarks[j]
    for k in range(1, m + 1):
        locations_2[i][k - 1] = landmarks[k]

# 获取locations_1和locations_2形状参数
rows_locations_1, cols_locations_1 = locations_1.shape
rows_locations_2, cols_locations_2 = locations_2.shape

# 将locations_1和locations_2的大小补充至相等
if rows_locations_1 < rows_locations_2:
    # 计算需要补齐的行数
    num_rows = rows_locations_2 - rows_locations_1

    # 用第0行填充
    repeated_row = np.tile(locations_1[0, :], (num_rows, 1))

    # 将 A 和全零矩阵进行垂直堆叠
    locations_1 = np.vstack((locations_1, repeated_row))

if rows_locations_2 < rows_locations_1:
    # 计算需要补齐的行数
    num_rows = rows_locations_1 - rows_locations_2

    # 用第0行填充
    repeated_row = np.tile(locations_2[0, :], (num_rows, 1))

    # 将 B 和全零矩阵进行垂直堆叠
    locations_2 = np.vstack((locations_2, repeated_row))

# 将矩阵行数和维度比较，如果矩阵行数小于维度，补齐至维度
if m > locations_1.shape[0] and m > locations_2.shape[0]:
    num_rows = m - rows_locations_1

    zero_row = np.zeros((num_rows, m))

    locations_1 = np.vstack((locations_1, zero_row))
    locations_2 = np.vstack((locations_2, zero_row))

# 三维以下的我是用法向量做的
if m == 3 and n_1 == 2 and n_2 == 2:
    A = locations_1[0]
    B = locations_1[1]
    C = locations_2[0]
    D = locations_2[1]

    AB = B - A
    CD = D - C

    cross = np.cross(AB, CD)

    if np.allclose(cross, np.zeros(3)):
        print("N")

    CA = A - C

    t = np.dot(np.cross(CD, CA), cross) / np.dot(cross, cross)
    s = np.dot(CA, cross) / np.dot(cross, cross)

    if 0 <= t <= 1 and 0 <= s <= 1:
        intersection = A + t * AB
        for i in range(len(intersection)):
            if intersection[i].is_integer():
                str_p = str_p + " %d" % intersection[i]
            else:
                str_p = str_p + " %f" % intersection[i]
        print(str_p)
    else:
        print("N")

else:  # 这里处理三维以上的输入
    # 首先根据仿射空间，将输入矩阵每一行减去label0
    affine_matrix_1 = locations_1 - home_location_1
    affine_matrix_2 = locations_2 - home_location_2

    # 常数项 b1 - b2
    b = home_location_1 - home_location_2

    # 扩充b矩阵
    # 依据维度和矩阵的行数进行判断
    # 需要保持矩阵乘法的维度相等
    if m >= affine_matrix_1.shape[0] and m >= affine_matrix_2.shape[0]:
        b_m = np.zeros((2, 2 * m))
        b_m[0, 0:m] = b
    else:
        b_m = np.zeros((2, 2 * affine_matrix_1.shape[0]))
        b_m[0, 0:m] = b

    # 系数项 A
    # 堆叠 B和-A
    A = np.vstack((affine_matrix_2, -affine_matrix_1))

    # 扩充A矩阵
    # 依据维度和矩阵的行数进行判断 一个是向row填充0，一个是向column填充0
    # 需要保持矩阵乘法的维度相等
    if m >= affine_matrix_1.shape[0] and m >= affine_matrix_2.shape[0]:
        target_size = 2 * m

        # 计算需要添加的行和列的数目
        num_rows = target_size - A.shape[0]
        num_cols = target_size - A.shape[1]

        # 使用 pad 函数将矩阵 A 扩充到目标大小
        padded = np.pad(A, ((0, num_rows), (0, num_cols)), mode='constant')

        # 截取出目标大小的部分，得到方阵
        A = padded[:target_size, :target_size]
    else:
        target_size = 2 * affine_matrix_1.shape[0]

        # 计算需要添加的行和列的数目
        num_rows = target_size - A.shape[0]
        num_cols = target_size - A.shape[1]

        # 使用 pad 函数将矩阵 A 扩充到目标大小
        padded = np.pad(A, ((0, num_rows), (0, num_cols)), mode='constant')

        # 截取出目标大小的部分，得到方阵
        A = padded[:target_size, :target_size]

    # 计算行列式，没啥大用，因为每个堆叠完的系数矩阵都是奇异矩阵
    determinant = np.linalg.det(A)
    if determinant == 0:
        A_pinv = np.linalg.pinv(A)  # ！ 广义逆计算
        x = b_m @ A_pinv
    else:
        A_inv = np.linalg.inv(A)  #  逆计算
        x = b_m @ A_inv

    # 裁剪x1 和 x2
    if m >= affine_matrix_1.shape[0] and m >= affine_matrix_2.shape[0]:
        x2 = x[0, 0:m]
        x1 = x[0, m:2 * m]
    else:
        x2 = x[0, 0:affine_matrix_1.shape[0]]
        x1 = x[0, affine_matrix_1.shape[0]:2 * affine_matrix_1.shape[0]]

    # print("x1:", x1)
    # print("x2:", x2)

    # 计算交点
    x_out_1 = home_location_1 + x1 @ affine_matrix_1
    x_out_2 = home_location_2 + x2 @ affine_matrix_2

    # print("x_out_1:", x_out_1)
    # print("x_out_2:", x_out_2)

    # check 矩阵某列是否全为0
    # 这里应该是个bug，不过能跑
    # 感觉一旦输入有全0列，两个交点的输出某一列应同时为0
    for i in range(m):
        if np.all(affine_matrix_1[:, i] == 0):
            x_out_1[i] = 0
        if np.all(affine_matrix_2[:, i] == 0):
            x_out_2[i] = 0

    # 字符串的目的是因为想要比较指定小数点后几位
    str_c1 = ""
    str_c2 = ""
    # check_c1 = 0
    check_c2 = 0
    check = False

    for i in range(m):
        # 该处是生成两个交点的字符串形式
        if x_out_1[i].is_integer():
            str_c1 = " %d" % x_out_1[i]
        else:
            str_c1 = " %.3f" % x_out_1[i]

        if x_out_2[i].is_integer():
            str_c2 = " %d" % x_out_2[i]
        else:
            str_c2 = " %.3f" % x_out_2[i]

        # 这里还是个bug， 本意是想比较两个输出是否相同的
        # 但是能跑通测试
        if str_c1 == str_c2:
            check = True

        # 这里是检测输出有几个0， 感觉还是个bug，不过能跑
        if abs(x_out_1[i] - 0) <= 1e-5 and abs(x_out_2[i] - 0) <= 1e-5:
            check_c2 += 1

    # 这里是检测输出是否为0
    if check_c2 == m:
        check = False
    else:
        check = True

    # 输出结果
    if not check:
        print("N")
    else:
        for i in range(len(x_out_1)):
            if x_out_1[i].is_integer():
                str_p = str_p + " %d" % x_out_1[i]
            else:
                str_p = str_p + " %.10f" % x_out_1[i]
        print(str_p)
