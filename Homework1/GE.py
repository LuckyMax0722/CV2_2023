import numpy as np


n = int(input())

matrix = np.zeros((n, n))  # 创建空矩阵

for i in range(n):  # 读取矩阵信息
    row = input().split()
    for j in range(n):
        matrix[i][j] = row[j]

augmented = np.concatenate((matrix, np.eye(n)), axis=1)  # A + I

zero_row = True

str_M = ""
str_A = ""
str_S = ""
str_E = ""
str_MI = ""
str_MA = ""
i = 0
not_change_more = False
c = 1

while i <= n-1:
    pivot = augmented[i][i]

    if pivot != 1 and pivot != 0:
        augmented[i] /= pivot
        str_M = "M %d %.20f" % (i, 1 / pivot)
        str_E = str_E + str_M + "\n"

        j = 0
        while j <= n - 1:
            if j == i:
                j += 1
            else:
                factor = augmented[j][i]
                if factor != 0:
                    augmented[j] += -factor * augmented[i]

                    if factor.is_integer():
                        str_A = "A %d %d %d" % (j, i, -factor)
                        str_E = str_E + str_A + "\n"
                    else:
                        str_A = "A %d %d %.20f" % (j, i, -factor)
                        str_E = str_E + str_A + "\n"
                    j += 1
                else:
                    j += 1
        i += 1

    elif pivot == 1 and pivot != 0:
        l = 0
        while l <= n-1:
            if l == i:
                l += 1
            elif augmented[l][i] != 0 and l != i:
                factor = augmented[l][i]
                augmented[l] += -factor * augmented[i]
                if factor.is_integer():
                    str_A = "A %d %d %d" % (l, i, -factor)
                    str_E = str_E + str_A + "\n"
                else:
                    str_A = "A %d %d %.20f" % (l, i, -factor)
                    str_E = str_E + str_A + "\n"
                l += 1
            else:
                l += 1
        i = i + 1

    elif pivot == 1 and pivot == 0:
        i = i + 1

    elif pivot == 0 and not_change_more is False:
        changed = False
        k = i + 1
        tmp = []

        while k <= n-1 and changed is False:
            if augmented[k][i] != 0:
                tmp = augmented[i].copy()
                augmented[i] = augmented[k].copy()
                augmented[k] = tmp.copy()
                changed = True
                str_S = "S %d %d" % (i, k)
                str_E = str_E + str_S + "\n"
                # print("S %d %d" % (i, k))
            else:
                k = k + 1

        if changed is False:
            i = i + 1

i = 0
while i <= n - c:
    if np.all(augmented[i, 0:n] == 0):
        k = i
        while k <= n - 2:
            tmp = augmented[k + 1].copy()
            augmented[k + 1] = augmented[k].copy()
            augmented[k] = tmp.copy()
            str_S = "S %d %d" % (k, k+1)
            str_E = str_E + str_S + "\n"
            k += 1
        zero_row = False
        c += 1
    else:
        i = i + 1

if zero_row:
    de_augmented = augmented[:, n:]
    str_E = str_E + "SOLUTION" + "\n"
    for i in range(n):
        for j in range(n):
            if de_augmented[j][i].is_integer():
                str_MI = "%d" % de_augmented[i][j]
                str_MA = str_MA + str_MI + " "
            else:
                str_MI = "%.20f" % de_augmented[i][j]
                str_MA = str_MA + str_MI + " "
        str_MA = str_MA + "\n"
    str_E = str_E + str_MA
else:
    str_E = str_E + "DEGENERATE"
print(str_E)

