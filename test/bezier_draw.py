import matplotlib.pyplot as plt 
import numpy as np

P1, P2, P3, P4 = np.array([0,0]), np.array([0.5,4]), np.array([1,4]), np.array([1.5, 4])
#P1, P2, P3, P4 = -5, 0.3, 0.5, 100
# vec: P1 -> P2, P4 -> P3
def Bezier(t):
    return P1 * ((1 - t) ** 3) + 3 * P2 * t * ((1 - t) ** 2) + 3 * P3 * (t ** 2) * (1 - t) + P4 * (t ** 3) 

x = np.arange(0, 1, 0.01)
y = np.array([Bezier(i) for i in x])
plt.scatter(y[:, 0], y[:, 1])
plt.show()