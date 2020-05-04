import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import math as math
import time

start = time.time()

X = pd.read_csv(r'F:\UNIBUC\3AN\Sem2\TO\a9aTrain.txt', header=None, delimiter=' ');

X[13].fillna(0, inplace=True)
X[14].fillna(0, inplace=True)
X[15].fillna(0, inplace=True)

X = np.array(X);
y = X[:, 0];
X = X[:, 1:];
L1 = True;
# or L2 = False

[d, n] = X.shape;

# Penalty parameter
C = 1;

# Maximum iterations
tmax = 1000;

if L1 == True:
    U = C;
    Dii = 0;
else:
    U = 999999;
    Dii = 1 / 2*C;

#
alpha = np.zeros(y.shape);
w = np.zeros((d, 1))

Qii = X*X + Dii;
Qii = Qii.sum(0);
Qii = np.nan_to_num(Qii)

for t in range(0, tmax):
    
    for i in range(1, n):
        
        g = np.multiply((np.dot(X[:, i], w))[0], y[i]) - 1 + Dii * alpha[i];
        
        g = np.nan_to_num(g)
        
        if alpha[i] == 0:
            g = np.minimum(g, 0);
        if alpha[i] == U:
            g = np.maximum(g, 0);        
            
        if np.absolute(g) != 0:
            
            if Qii[i] == 0:
                alpha_new = np.minimum(0, U)
            else:
                alpha_new = np.minimum(np.maximum(alpha[i] - g / Qii[i], 0), U);
            
            rr = np.transpose(w) + (np.multiply((alpha_new - alpha[i]), y[i])) * X[:, i];
            w = np.transpose(rr);
            alpha[i] = alpha_new;

end = time.time()
print(end - start)