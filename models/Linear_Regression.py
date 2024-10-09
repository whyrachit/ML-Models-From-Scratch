import numpy as np

class Linear_Regression:

    def __init__(self):

        self.intercept=None
        self.coeff=None
    
    def fit(self,X,y):

        ones=np.ones((len(X),1))
        X=np.concatenate((ones,X),axis=1)
    
        XT=X.T
        XTX=XT.dot(X)
        XTX_inverse=np.linalg.inv(XTX)
        XTy=XT.dot(y)
        self.coeff=XTX_inverse.dot(XTy)
    
    def predict(self,X):

        ones=np.ones((len(X),1))
        X=np.concatenate((ones,X),axis=1)

        return X.dot(self.coeff)
    
    def Rsquared(self, X, y):

        y_pred = self.predict(X) 
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_total)