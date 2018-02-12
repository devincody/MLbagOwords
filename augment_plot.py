#augment_plot.py

import matplotlib.pyplot as plt
import numpy as np
x = np.array([15000, 20000, 40000, 60000, 80000, 100000, 120000])
y = np.array([3.4082, 3.3978,3.3868,3.3892,3.3922,3.3776,3.3892])
y/=4
plt.plot(x,y)
plt.ylabel("Accuracy")
plt.xlabel("Total Training Examples")
plt.title("4-fold Validated Augmented Training")
plt.show()