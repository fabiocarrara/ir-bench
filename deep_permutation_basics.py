import numpy as np


def crelu_perm(a, k):
    permutation = np.argsort(a)[::-1] + 1  # 1-indexed permutation
    inverse_permutation = np.argsort(permutation) + 1  # 1-indexed inverse permutation
    complemented_inverse_permutation = len(a) + 1 - inverse_permutation  # term frequencies
    truncated_complemented_inverse_permutation = np.maximum(complemented_inverse_permutation - (k+1), 0)
    return permutation, inverse_permutation, complemented_inverse_permutation, truncated_complemented_inverse_permutation


k = 2

q = np.array([-0.4, 0.2, 0.7, -0.1, 0.5])
x = np.array([0.1, 0.3, 0.4, -0.15, 0.2])
y = np.array([0.0, -0.8, 0.7, 0.9, 1.2])

pq, ipq, cipq, tipq = crelu_perm(q, k)
px, ipx, cipx, tipx = crelu_perm(x, k)
py, ipy, cipy, tipy = crelu_perm(y, k)

print(q, x, y)
print(pq, px, py)
print(ipq, ipx, ipy)
print(cipq, cipx, cipy)
print(tipq, tipx, tipy)


