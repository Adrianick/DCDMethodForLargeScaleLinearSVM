import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import time

start = time.time()

X = pd.read_csv(r'F:\UNIBUC\3AN\Sem2\TO\a9aTrain.txt', header=None, delimiter=' ');

X[13].fillna(0, inplace=True)
X[14].fillna(0, inplace=True)
X[15].fillna(0, inplace=True)
#X = X.dropna();

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

# Epsilon
eps = 1e-3;


if L1 == True:
    U = C;
    Dii = 0;
else:
    U = 999999;
    Dii = 1 / 2*C;

#
alpha = np.zeros(n); # y.shape
w = np.zeros((n, 1)) # d

Qii = X*X + Dii;
Qii = Qii.sum(0);
Qii = np.nan_to_num(Qii)

Mb = np.inf;
mb = -np.inf;

A = np.array(list(range(1, n)));


X = np.nan_to_num(X)
y = np.nan_to_num(y)

for t in range(0, tmax):
    
    M = -np.inf;
    m = np.inf;
    
    for l in range(0, A.shape[0]):
        A_changed = False;
        
        i = A[l]        
        
        g = np.multiply((np.dot(X[i], w))[0], y[i]) - 1 + Dii * alpha[i];
        
        g = np.nan_to_num(g)
        
        if alpha[i] == 0:
            g = np.minimum(g, 0);
            
           
            
            if g > Mb:
#                A[l] = 0;
#                A = np.nan_to_num(A)
                A_changed = True;
                continue;
            
        if alpha[i] == U:
            g = np.maximum(g, 0);
            
              
            if g < mb:
#                A[l] = 0;
#                A = np.nan_to_num(A)
                A_changed = True;
                continue;
            
        M = np.maximum(M, g);
        m = np.minimum(m, g);
        
        if np.absolute(g) != 0:
            
            if Qii[i] == 0:
                alpha_new = np.minimum(0, U)
            else:
                alpha_new = np.minimum(np.maximum(alpha[i] - g / Qii[i], 0), U);
            
            rr = np.transpose(w) + (np.multiply((alpha_new - alpha[i]), y[i])) * X[i];
            w = np.transpose(rr);
            alpha[i] = alpha_new;
            
        
        if M - m < eps:
            if A_changed == False:
                break;
            else:
#                A = np.array(list(range(1, n+1)));
                Mb = np.inf;
                mb = -np.inf;
        
        if M <= 0:
            Mb = np.inf;
        else:
            Mb = M;
            
        if m >= 0:
            mb = -np.inf;
        else:
            mb = m;

end = time.time()
print(end - start)

Xtest = pd.read_csv(r'F:\UNIBUC\3AN\Sem2\TO\a9aF.txt', header=None, delimiter=' ');

Xtest[13].fillna(0, inplace=True)
Xtest[14].fillna(0, inplace=True)
Xtest[15].fillna(0, inplace=True)
#Xtest = Xtest.dropna()


Xtest = np.array(Xtest);
ytest = Xtest[:, 0];
Xtest = Xtest[:, 1:];

Xmeans = np.zeros(w.shape[0]);
for i in range(0, w.shape[0]):
    Xmeans[i] += np.mean(X[i])


X = np.nan_to_num(X)
y = np.nan_to_num(y)

Xtest = np.nan_to_num(Xtest)
ytest = np.nan_to_num(ytest)


#

#
#y_p = clf.predict(Xtest)

#
#w1 = np.zeros(w.shape[0]);
#w2 = np.zeros(w.shape[0]);
#
#
#

y_p = np.array([])
for i in range(0, Xtest.shape[0]):
    tt = np.reshape(Xtest[i], (1, 15));
    p = np.sign(np.dot(tt, w));
    y_p = np.append(y_p, p);



print('Accuracy = ', np.sum(y_p == ytest) / ytest.shape[0]);



import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import time

start = time.time()

from sklearn import svm

X = np.nan_to_num(X)
y = np.nan_to_num(y)

clf = svm.SVC()
clf.fit(X, y)



end = time.time()
print(end - start)

yyp = clf.predict(Xtest)
print('Accuracy Classic = ', np.sum(yyp == ytest) / ytest.shape[0]);
#
#mean1 = np.array([0, 2])
#mean2 = np.array([2, 0])
#cov = np.array([[1.5, 1.0], [1.0, 1.5]])
#X1 = np.random.multivariate_normal(mean1, cov, 100)
#y1 = np.ones(len(X1))
#X2 = np.random.multivariate_normal(mean2, cov, 100)
#y2 = np.ones(len(X2)) * -1
#
#X1_train = X1[:90]
#y1_train = y1[:90]
#X2_train = X2[:90]
#y2_train = y2[:90]
#X_train = np.vstack((X1_train, X2_train))
#y_train = np.hstack((y1_train, y2_train))
#
#X1_test = X1[90:]
#y1_test = y1[90:]
#X2_test = X2[90:]
#y2_test = y2[90:]
#X_test = np.vstack((X1_test, X2_test))
#y_test = np.hstack((y1_test, y2_test))
#
#
#ww = w[:20]
#
#wwt = np.transpose(ww)
#
#yp1 = np.sign(np.dot(X_test, wwt));
#
#
#xs = [];
#xd = [];
#for i in range(0, y.shape[0]):
#    if y[i] == -1:
#        xs.append(X[i]);
#    else:
#        xd.append(X[i]);


#from sklearn.datasets.samples_generator import make_blobs 
#  
## plotting scatters  
#plot.scatter(xs[:, 0], xs[:, 1], c=xd, s=50, cmap='spring'); 
#plot.show()  
#










