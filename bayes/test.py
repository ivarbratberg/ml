import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import slogdet, det, solve
def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights 
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices
    
    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0] # number of objects
    C = pi.shape[0] # number of clusters
    d = X.shape[1] # dimension of each object
    gamma = np.zeros((N, C)) # distribution q(T)
    print("Hei")
    ### YOUR CODE HERE
    for c in np.arange(0,C):
      for ix in np.arange(0,N):
        x = X[ix,:]
        xc = x - mu[c,:]
        sigmac = sigma[c,:,:]
        sigmacInv_xc = solve(a=sigmac, b= xc)
        exp_arg_c = -0.5*np.dot(xc , sigmacInv_xc)
        acc = 0.0
        for d in np.arange(0,C):
          xd = x - mu[d,:]
          sigmad = sigma[d,:,:]
          sigmadInv_xd = solve(a=sigmad, b= xd)
          exp_arg_d = -0.5*np.dot(xd,  sigmadInv_xd)
          exp_diff = exp_arg_d - exp_arg_c
          acc = acc +  (pi[d]/pi[c]) * np.sqrt(det(sigmad)/det(sigmac))*np.exp(exp_diff)          
        gamma[ix,c] = 1/acc      
        
    
    return gamma

def M_step(X, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    
    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    pi = np.zeros([C])
    mu = np.zeros([C, d])
    sigma = np.zeros([C, d, d])
    
    for c in np.arange(0,C):
      mu_nominator = np.zeros([1,d])
      sigma_nominator = np.zeros([1,d])
      qc_sum = 0.0
      for i in np.arange(0,N):
        x_vec = X[i,].reshape([1,d])
        mu_nominator = mu_nominator + gamma[i,c]*x_vec
        sigma_nominator = sigma_nominator + gamma[i,c]*np.transpose(x_vec)*x_vec
        qc_sum = qc_sum + gamma[i,c]
      pi[c] = qc_sum/N
      mu[c,] = mu_nominator/qc_sum
      sigma[c,:,:] = sigma_nominator/qc_sum

    return pi, mu, sigma

def compute_vlb(X, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)  
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    
    Returns value of variational lower bound
    """
    N = X.shape[0] # number of objects
    C = gamma.shape[1] # number of clusters
    d = X.shape[1] # dimension of each object

    loss = 0
    for c in np.arange(0,C):
      sig = sigma[c,:,:]
      muh = mu[c,:].reshape([d])
      pih = pi[c]
      var=multivariate_normal(mean=muh,cov=sig)
      for i in np.arange(0,N):
        q = gamma[i,c]
        x= X[i,]
        if (pih == 0 or var.pdf(x) == 0 or q == 0):
              print("Zerio in log")
        loss = loss + q*(np.log(pih)+np.log(var.pdf(x)) - np.log(q))          
          

    return loss    

def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=10):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.
    
    X: (N, d), data points
    C: int, number of clusters
    '''
    N = X.shape[0] # number of objects
    d = X.shape[1] # dimension of each object
    best_loss = None
    best_pi = None
    best_mu = None
    best_sigma = None
    sigma = np.zeros([C,d,d])
    mu = np.zeros([C,d])
    x_max = np.amax(X,0)
    x_min = np.amin(X,0)

    # We need to restart
    for _ in range(restarts):
        # Prepare mu and sigma for all classes
        for c in range(C):
            sigma[c,:,:] = np.identity(d)
            mu[c,:] = np.random.uniform(x_min,x_max)   
        lims = np.random.uniform(0,1,C-1)
        lims.sort()
        lims=np.append(lims,1)
        lims=np.insert(lims,0,0)
        pi = np.diff(lims)
        candidate_best_loss = None
        try:
            prev_loss = None
            # Iterate the EM
            print("New restart ---------")
            for _ in range(max_iter):                
                gamma = E_step(X,pi,mu,sigma)
                pi, mu, sigma = M_step(X, gamma)
                loss = compute_vlb(X,pi,mu,sigma, gamma)
                if prev_loss == None:
                    prev_loss = loss
                    next                                
                loss_diff = prev_loss - loss
                candidate_best_loss = loss_diff/prev_loss
                print("Loss " + str(loss) + " " + str(candidate_best_loss))
                if (loss > -1245.5694233440036):
                  print("debug ")

                
            # Is this global best loss ?
            if  best_loss == None or best_loss > candidate_best_loss:
                best_loss = candidate_best_loss
                best_pi = pi
                best_mu = mu
                best_sigma = sigma

            if candidate_best_loss < rtol:
                    break

        except np.linalg.LinAlgError:
            print("Singular matrix: components collapsed")
            pass

    return best_loss, best_pi, best_mu, best_sigma

