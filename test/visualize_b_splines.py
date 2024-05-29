import matplotlib.pyplot as plt 
import numpy as np

t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

P = np.array([[0, 0], [4, 0], [0, 4], [4, 4]])

def B_spline_basis(x, i, p):
    # return B_i_p(x)
    if (p == 0):
        if t[i] <= x and x <= t[i + 1]:
            return 1;
        else:
            return 0;
    else:
        return B_spline_basis(x, i, p - 1) * (x - t[i]) / (t[i + p] - t[i]) + B_spline_basis(x, i + 1, p - 1) * (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]);
    
def B_spline(x, i):
    # return B_i(x)
    return B_spline_basis(x, i, 3);

# x = np.arange(4, 10, 0.1)
# y = np.array([B_spline(i, 0) for i in x])
# z = np.array([B_spline(i, 1) for i in x])
# u = np.array([B_spline(i, 2) for i in x])
# v = np.array([B_spline(i, 3) for i in x])
# plt.plot(x, y);
# plt.plot(x, z);
# plt.plot(x, u);
# plt.plot(x, v);
# plt.plot(x, y * 2 - 2 * z + u + v)
tim = np.arange(3, 4, 0.01)
x = [B_spline(tim_p, 0) * P[0, 0] + B_spline(tim_p, 1) * P[1, 0] + B_spline(tim_p, 2) * P[2, 0] + B_spline(tim_p, 3) * P[3, 0] for tim_p in tim]
y = [B_spline(tim_p, 0) * P[0, 1] + B_spline(tim_p, 1) * P[1, 1] + B_spline(tim_p, 2) * P[2, 1] + B_spline(tim_p, 3) * P[3, 1] for tim_p in tim]
plt.scatter(x, y)
plt.scatter(P[:, 0], P[:, 1], c="r")
plt.show()