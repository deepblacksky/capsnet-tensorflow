import matplotlib.pyplot as plt
import numpy as np

step, acc = np.loadtxt('accuracy.csv', delimiter=',', unpack=True, skiprows=1)
plt.figure()
plt.plot(step, acc, label="epoch:30", color="blue", linewidth=2)
plt.xlabel("step")
plt.ylabel("accuracy")
plt.title("Capsnet Result")
plt.legend()
plt.savefig('accuracy.png')
plt.show()
