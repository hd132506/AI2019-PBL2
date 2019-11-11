import numpy as np
from adaline import AdalineGD

agd = AdalineGD(eta=0.001)

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 1], [7, 8, 8], [7, 7, 7]])
Y = np.array([-1, -1, 1, -1, 1, 1])


agd.fit(X, Y)

print(agd.predict(X))
