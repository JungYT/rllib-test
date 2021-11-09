import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0, 1, 10)
y = x ** 2

fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot(x, y)
fig.savefig('.', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot(x, y)
ax.set_ylabel('REAL')
fig.savefig('.', bbox_inches='tight')
