import time
import numpy as np

from scripts.env.orbitx import rv_from_r0v0


rv0 = np.array([
    7000.0, -12124.0, 0.0,
    2.6679, 4.621, 0.0
])

t = 3600.0

t_start = time.time()

rv1 = rv_from_r0v0(rv0, t)

t_end = time.time()

print(t_end - t_start)