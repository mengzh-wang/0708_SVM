import numpy as np
from matplotlib import pyplot as plt
import time
from qpsolvers import solve_qp
import math

"""----------------------结果统计----------------------"""


def statistic(w, b, xin, yin, xout, yout):
    wrong_cases_train = 0
    wrong_cases_test = 0
    nin = len(xin)
    nout = len(xout)
    for j in range(nin):
        if (np.dot(w, xin[j].T) + b) * yin[j] <= 0:
            wrong_cases_train += 1
    wrong_rate_train = wrong_cases_train / nin

    for j in range(nout):
        if (np.dot(w, xout[j].T) + b) * yout[j] <= 0:
            wrong_cases_test += 1
    wrong_rate_test = wrong_cases_test / nout

    print("训练集正确率=", 1 - wrong_rate_train)
    print("测试集正确率=", 1 - wrong_rate_test)

    return 0


def statistic_kernel(sv_indexes, alpha, b, xin, yin, xout, yout, method):
    wrong_cases_train = 0
    wrong_cases_test = 0
    nin = len(xin)
    nout = len(xout)
    n_sv = np.shape(sv_indexes)[0]
    for j in range(nin):
        temp = 0
        for k in range(n_sv):
            temp = temp + alpha[sv_indexes[k]] * yin[sv_indexes[k]] * kernel_filling(xin[sv_indexes[k]], xin[j],
                                                                                     method)
        if (temp + b) * yin[j] <= 0:
            wrong_cases_train += 1
    wrong_rate_train = wrong_cases_train / nin

    for j in range(nout):
        temp = 0
        for k in range(n_sv):
            temp = temp + alpha[sv_indexes[k]] * yin[sv_indexes[k]] * kernel_filling(xin[sv_indexes[k]], xout[j],
                                                                                     method)
        if (temp + b) * yout[j] <= 0:
            wrong_cases_test += 1
    wrong_rate_test = wrong_cases_test / nout

    print("训练集正确率=", 1 - wrong_rate_train)
    print("测试集正确率=", 1 - wrong_rate_test)

    return 0


"""----------------------Primal-SVM----------------------"""


def primal_svm(xin, yin):
    x = np.copy(xin)  # 320*2
    y = np.copy(yin)  # 320*1
    nx, dx = np.shape(x)
    q = np.eye(dx + 1)
    q[0][0] = 0
    p = np.zeros([dx + 1, 1])
    a = np.ones([nx, 1])  # 320*3
    a = np.hstack([a, x])
    for j in range(nx):
        a[j] = a[j] * y[j]
    c = np.ones([nx, 1])  # 320*1

    a = -1 * a
    c = -5 * c

    solution = solve_qp(q, p, G=a, h=c, A=None, b=None, lb=None, ub=None, solver="scs")
    b = solution[0]
    w = solution[1:3]
    return b, w


"""----------------------Dual-SVM----------------------"""


def find_sv(q, nx, yin):
    p = -1 * np.ones([nx, 1])
    a = np.eye(nx)
    c = np.zeros([nx, 1])
    a = -1 * a
    c = -1 * c
    r = yin.T[0]
    v = np.zeros(1)

    alpha = solve_qp(q, p, G=a, h=c, A=r, b=v, lb=None, ub=None, solver="scs")
    alpha[alpha < 0.0001] = 0

    sv_indexes = np.nonzero(alpha)
    sv_indexes = np.array(sv_indexes[0])

    return alpha, sv_indexes


def dual_svm(xin, yin):
    x = np.copy(xin)
    y = np.copy(yin)
    nx, dx = np.shape(x)
    q = np.zeros([nx, nx])
    for j in range(nx):
        for k in range(nx):
            q[j][k] = y[j] * y[k] * np.dot(x[j], x[k].T)

    alpha, sv_indexes = find_sv(q, nx, y)
    w = np.zeros(dx)
    b = 0

    n_sv = np.shape(sv_indexes)[0]
    for j in range(nx):
        w = w + alpha[j] * y[j] * x[j]
    for j in range(n_sv):
        b = b + y[sv_indexes[j]] - np.dot(w, x[sv_indexes[j]].T)
    b = sum(b) / n_sv
    return b, w


"""----------------------Kernel-SVM----------------------"""


def kernel_filling(x1,x2,method):
    match method:
        case 'quartic_polynomial':
            zeta = 1
            gamma = 1
            return (zeta + gamma * np.dot(x1, x2.T)) ** 4
        case 'gaussian':
            gamma = 1
            norm = np.linalg.norm(x1 - x2)
            return math.exp(-gamma * norm ** 2)


def kernel_svm(xin, yin, method):
    x = np.copy(xin)
    y = np.copy(yin)

    # 归一化
    '''min = np.min(x, axis=0)
    max = np.max(x, axis=0)
    diff = max - min
    x[:, 0] = (x[:, 0] - min[0]) / diff[0]
    x[:, 1] = (x[:, 1] - min[1]) / diff[1]'''

    nx, dx = np.shape(x)
    q = np.zeros([nx, nx])
    for j in range(nx):
        for k in range(nx):
            q[j][k] = y[j] * y[k] * kernel_filling(x[j], x[k], method)
    alpha, sv_indexes = find_sv(q, nx, y)
    n_sv = np.shape(sv_indexes)[0]

    b = 0
    for j in range(n_sv):
        temp = 0
        for k in range(n_sv):
            temp = temp + alpha[sv_indexes[k]] * y[sv_indexes[k]] * kernel_filling(x[sv_indexes[k]], x[sv_indexes[j]], method)
        b = b + y[sv_indexes[j]] - temp
    b = sum(b) / n_sv
    w = 0

    for j in range(nx):
        w = w + alpha[j] * y[j] * x[j]

    return sv_indexes, alpha, b


"""----------------------数据集初始化----------------------"""

# 数据分布与规模
u1 = [-5, 0]
s1 = [[1, 0], [0, 1]]
u2 = [0, 5]
s2 = [[1, 0], [0, 1]]
n = 200
train_rate = 0.8
n_train = int(n * train_rate)
n_test = n - n_train
# 数据填充
x1 = np.empty([n, 2])  # A
x2 = np.empty([n, 2])  # B
x_train = np.empty([n_train * 2, 2])  # 320
x_test = np.empty([n_test * 2, 2])  # 80

for i in range(n):  # 200
    x1[i] = np.random.multivariate_normal(u1, s1)
    x2[i] = np.random.multivariate_normal(u2, s2)

for i in range(n_train):  # 160
    x_train[i] = x1[i]  # A
    x_train[n_train + i] = x2[i]  # B
for i in range(n_test):  # 40
    x_test[i] = x1[i]  # A
    x_test[n_test + i] = x2[i]  # B

y_train = np.empty([n_train * 2, 1])
for i in range(n_train):
    y_train[i] = 1
    y_train[n_train + i] = -1
y_test = np.empty([n_test * 2, 1])
for i in range(n_test):
    y_test[i] = 1
    y_test[n_test + i] = -1


"""----------------------代码运行----------------------"""

time_start = time.time()
b, w = primal_svm(x_train, y_train)
time_end = time.time()
time_primal_svm = time_end - time_start

time_start = time.time()
b_dual_svm, w_dual_svm = dual_svm(x_train, y_train)
time_end = time.time()
time_dual_svm = time_end - time_start

time_start = time.time()
sv_indexes_quartic, alpha_quartic, b_quartic = kernel_svm(x_train, y_train, method='quartic_polynomial')
time_end = time.time()
time_quartic_svm = time_end - time_start

time_start = time.time()
sv_indexes_gaussian, alpha_gaussian, b_gaussian = kernel_svm(x_train, y_train, method='gaussian')
time_end = time.time()
time_gaussian_svm = time_end - time_start

x_min = min(min(x1[:, 0]), min(x2[:, 0]))
x_max = max(max(x1[:, 0]), max(x2[:, 0]))
y_min = min(min(x1[:, 1]), min(x2[:, 1]))
y_max = max(max(x1[:, 1]), max(x2[:, 1]))
xx = np.arange(x_min - 1, x_max + 1, 0.05)
yy = np.arange(y_min - 1, y_max + 1, 0.05)
A, B = np.meshgrid(xx, yy)
height, width = np.shape(A)


print("--------------Primal-SVM结果统计--------------")
print("w=", w)
print("b=", b)
statistic(w, b, x_train, y_train, x_test, y_test)
print("算法运行时间=", time_primal_svm, "s")

print("--------------Dual-SVM结果统计--------------")
print("w=", w_dual_svm)
print("b=", b_dual_svm)
statistic(w_dual_svm, b_dual_svm, x_train, y_train, x_test, y_test)
print("算法运行时间=", time_dual_svm, "s")

print("--------------Kernel-SVM(四次多项式)结果统计--------------")
statistic_kernel(sv_indexes_quartic, alpha_quartic, b_quartic, x_train, y_train, x_test, y_test, 'quartic_polynomial')
print("算法运行时间=", time_quartic_svm, "s")

print("--------------Kernel-SVM(高斯核函数)结果统计--------------")
statistic_kernel(sv_indexes_gaussian, alpha_gaussian, b_gaussian, x_train, y_train, x_test, y_test, 'gaussian')
print("算法运行时间=", time_gaussian_svm, "s")

print("----------------绘图中---------------")

plt.scatter(x1[:, 0], x1[:, 1], c='r')
plt.scatter(x2[:, 0], x2[:, 1], c='b')
plt.axline((0, -b / w[1]), slope=-w[0] / w[1], c='k', label='Primal-SVM')

plt.axline((0, -b_dual_svm / w_dual_svm[1]), slope=-w_dual_svm[0] / w_dual_svm[1], c='g', label='Dual-SVM')

gate_quartic = np.zeros([height, width])
gate_gaussian = np.zeros([height, width])
for j in range(height):
    for k in range(width):
        temp1 = 0
        temp2 = 0
        for l in sv_indexes_quartic:
            temp1 = temp1 + alpha_quartic[l] * y_train[l] * kernel_filling(x_train[l], np.array([A[j][k], B[j][k]]),
                                                                           'quartic_polynomial')
        for l in sv_indexes_gaussian:
            temp2 = temp2 + alpha_gaussian[l] * y_train[l] * kernel_filling(x_train[l], np.array([A[j][k], B[j][k]]),
                                                                           'gaussian')
        gate_quartic[j][k] = temp1 + b_quartic
        gate_gaussian[j][k] = temp2 + b_gaussian


plt.contour(A, B, gate_quartic, levels=[0], colors='c')
plt.contour(A, B, gate_gaussian, levels=[0], colors='m')

plt.xlim(x_min - 1, x_max + 1)
plt.ylim(y_min - 1, y_max + 1)
plt.legend()

plt.show()
print("--------------绘图输出完毕--------------")