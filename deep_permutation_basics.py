import numpy as np

k = 2+1

q = np.array([-0.4, 0.2, 0.7, -0.1, 0.5])
pq = np.argsort(q)[::-1] + 1 
ipq = np.argsort(pq) + 1
cipq = len(q) + 1 - ipq
tipq = np.maximum(cipq - k, 0)

x = np.array([0.1, 0.3, 0.4, -0.15, 0.2])
px = np.argsort(x)[::-1] + 1
ipx = np.argsort(px) + 1
cipx = len(x) + 1 - ipx
tipx = np.maximum(cipx - k, 0)

y = np.array([0.0, -0.8, 0.7, 0.9, 1.2])
py = np.argsort(y)[::-1] + 1 
ipy = np.argsort(py) + 1
cipy = len(y) + 1 - ipy
tipy = np.maximum(cipy - k, 0)

print(q, x, y)
print(pq, px, py)
print(ipq, ipx, ipy)
print(cipq, cipx, cipy)
print(tipq, tipx, tipy)


