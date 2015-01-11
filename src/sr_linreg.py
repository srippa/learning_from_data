import numpy as np  
import numpy.linalg as la

import scipy as sp
import scipy.stats as st

def t_critval(df,confint=0.95):
    ''' eturn critical value for a confidence interval for the t-distribution with df degrees of freedom '''
    t_rv        = st.t(df)
    alpha_h     = (1.-confint)/2.0
    critval     = t_rv.ppf(1.0-alpha_h)  # Taking the poaitive value of critval
    return critval

def t_pval(df,tstat):
    ''' Return probability of obtaiuning value of tstat or higher '''
    t_rv        = st.t(df)
    ts          = np.abs(tstat)
    pval        = t_rv.sf(ts)  
    return 2.0*pval

def f_critval(n,p,confint=0.95):
    ''' eturn critical value for a confidence interval for the t-distribution with df degrees of freedom '''
    f_rv        = st.f(p,n-p-1)
    alpha_h     = (1.-confint)/2.0
    critval     = f_rv.ppf(1.0-alpha_h)  # Taking the poaitive value of critval
    return critval

def f_pval(n,p,fstat):
    ''' Return probability of obtaiuning value of tstat or higher '''
    f_rv        = st.f(p,n-p-1)
    fs          = np.abs(fstat)
    pval        = f_rv.sf(fs)  
    return pval

# Residual sum of squares
def RSS(Y,ypred):
    return np.sum((Y-ypred)**2)

# Residual standard error - estimation for the Standard deviation of the error terms
# Also known as standard error
def RSE(Y,ypred,p):
    n = len(Y)
    return np.sqrt(RSS(Y,ypred)/float(n-p-1))

  
# Compute R^2 score for an estimator
def Rsquared(estimator,X,Y):
    ypred      = estimator.predict(X)
    ypred_mean = np.mean(ypred)
    rss_c      = RSS(Y,ypred)
    TSS        = np.sum((Y-ypred_mean)**2)
    Rsqr       = 1.0 - (rss_c/TSS)
    return Rsqr

# Compute F score. High F score indicates that mnultiple linear regression
# have at least one significant coefficient
def Fscore(estimator,X,Y):
    n          = X.shape[0]
    p          = X.shape[1]
    ypred      = estimator.predict(X)
    ypred_mean = np.mean(ypred)
    rss_c      = RSS(Y,ypred)
    TSS        = np.sum((Y-ypred_mean)**2)
    a          = (TSS-rss_c)/ float(p)
    b          = rss_c/float(n-p-1) 
    F          = a/b
    return F

# Compute R^2 score for an estimator
def AdjustedRsquared(rsqr,n,p):
    adj        = 1.0 - ( (1.0-rsqr)* (float(n)-1.)/(float(n)-float(p) ) )
    return adj


class MyLinearRegressor(object):
    def fit(self, Xin, y):
        """Fits estimator to data. """

        X = np.column_stack( (np.ones(len(Xin)), Xin) )
 
        self.n_          = Xin.shape[0]
        self.p_          = Xin.shape[1]
        
        self.X_          = X
        self.y_          = y
        self.X_norm_     = X.transpose().dot(X)   # Normal matrix
        self.X_norm_inv_ = la.inv(self.X_norm_)
        self.coef_       = self.X_norm_inv_.dot(X.transpose().dot(y))
        self.ypred_      = self.predict(Xin)

        # Quantities that measure global quality of fit
        #  RSE   - Global lack of fit
        #  R**2  - The proportion of variablity in Y that is explained by the model
        self.rse_        = RSE(self.y_,self.ypred_,self.p_+1)  # estimator for std of errors
        self.rsqr_       = Rsquared(self,Xin,y)
        self.F_          = Fscore(self,Xin,y)
        self.fprob_      = f_pval(self.n_,self.p_,self.F_)
 
        # Estimating quality of fit for different predictors
        self.stderr_     = np.array(self.StdError_())      
        self.critval_    = t_critval(self.n_-self.p_)
        self.zscore_     = self.coef_/self.stderr_              # How important is a coefficient
        self.pval_       = self.pval_()    

        # set state of ``self``
        return self
            
    def predict(self, X):
        """Predict response of ``X``. """
        # compute predictions ``pred``
        pred = X.dot(self.coef_[1:]) + self.coef_[0]
        return pred

    # Compute confidence inbterval for all predictors
    def confinterval(self,confint=0.95):
        CI = []
        critval = t_critval(self.n_-self.p_,confint)
        for (c,s) in zip(self.coef_,self.stderr_):
            ci_low  = c - critval*s
            ci_high = c + critval*s
            CI.append( (ci_low,ci_high))
        return CI

    # Compute p-values for all predictors. Each p-value is the probability that we get the value of the preduictor
    # that we got,under the hypothesis that the predictor value is zero (that is, the predictor is not
    # impportant)
    # If the value is small, e.g. < 0.005 then we reject the null hypothesis and conclude that the coefficient
    # is not neglible
    def pval_(self):
        PV = []
        for t in self.zscore_:
            PV.append(t_pval(self.n_-self.p_,t))
        return PV
            
    # Compute the standard errors for all predictors
    def StdError_(self):
        sigma_est     = self.rse_  # estimator for std of errors
        Xinvdiag      = np.diag(self.X_norm_inv_)
              
        SE  = []
        for v in Xinvdiag:
            SE_i = sigma_est*np.sqrt(v)
            SE.append(SE_i)

        return SE


 
class MyRidgeRegressor(object):
    def fit(self, X, Y, lamda,fit_intercept=True):
        """Fits estimator to data. """

        p                = X.shape[1] 
        Xlambda          = X.transpose().dot(X) + lamda*np.eye(p)
        inv              = la.inv(Xlambda)

        if fit_intercept:
            self.intercept_  = Y.mean()
        else:
            self.intercept_ = 0.0
        self.coef_       = inv.dot(X.transpose().dot(Y-self.intercept_))

        # set state of ``self``
        return self
            
    def predict(self, X):
        """Predict response of ``X``. """
        # compute predictions ``pred``
        pred = X.dot(self.coef_) + self.intercept_
        return pred

   
 
# Computing standard error (intercept and slope) for simple linear regression
def StdError(estimator,X,Y):
    ypred         = estimator.predict(X)
    sigma_est     = RSE(Y,ypred)
    sigma_sqr_est = sigma_est**2
    n     = len(X)
    print n,X.shape
    x          = X['TV']
    xmean      = np.mean(x)
    xmean_sqr  = xmean**2
    xt         = np.sum((x-xmean)**2)

    SE_B0_sqr1 = sigma_sqr_est*Xinvdiag[0];
    SE_B1_sqr1 = sigma_sqr_est*Xinvdiag[1];
    
    SE_B0_sqr  = sigma_sqr_est*( (1./float(n)) + (xmean_sqr/xt) )
    SE_B1_sqr  = sigma_sqr_est /xt

    print 'B0 ',SE_B0_sqr1,SE_B0_sqr
    print 'B1 ',SE_B1_sqr1,SE_B1_sqr
    
    return [np.sqrt(SE_B0_sqr),np.sqrt(SE_B1_sqr)]
  