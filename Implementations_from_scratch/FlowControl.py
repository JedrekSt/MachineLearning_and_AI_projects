import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series

########################################################################################


def add_to_class(Class_name):
    def inner(obj_name):
        setattr(Class_name,obj_name.__name__,obj_name)
    return inner

class NoKernel(Exception):
    def __init__(self,mess = None):
        self.message = mess
        super().__init__(mess)


########################################################################################


class Generate_Synth_Data_LR:
    def __init__(self,features_dim):
        self.fd = features_dim
    
    def fetch_data(self,num_of_features,seed = 42):
        raise NotImplementedError
    
    def get_w(self):
        raise NotImplementedError

    

@add_to_class(Generate_Synth_Data_LR)
def fetch_data(self,seed = 42):
    rng = np.random.default_rng(seed)
    dir_dict = {
        f"x{i}" : np.linspace(0,rng.integers(100),70) 
        for i in range(self.fd)
    }
    self.w_ = rng.uniform(size = self.fd + 1) * 100 
    target_data = {
        "y" : sum(self.w_[i] * dir_dict[f"x{i}"] for i in range(self.fd)) + self.w_[-1] 
    }
    noisify = lambda x_,sc : rng.normal(size = len(x_),loc = 0, scale = (np.max(x_) - np.min(x_)) / len(x_) * sc)
    for key in dir_dict.keys():
        dir_dict[key] += noisify(dir_dict[key],rng.integers(7) + 3)
    target_data["y"] += noisify(target_data["y"],4)
    return {"features" : DataFrame(dir_dict), "target" : Series(target_data)}

@add_to_class(Generate_Synth_Data_LR)
def get_w(self):
    return self.w_

########################################################################################

class Visualize_Linear_Data:
    def __init__(self,data_dict):
        self.data_dict = data_dict

    def get_meta_data(self):
        raise NotImplementedError

    def plot_raw_points(self,fig):
        raise NotImplementedError
    
    def correlations(self,data_dict):
        raise NotImplementedError
    
@add_to_class(Visualize_Linear_Data)
def get_meta_data(self):
    ln = len(self.data_dict["features"].keys())
    aa = np.sqrt(ln)
    return {
        "fn" : ln,
        "frame_size" : int(aa),
        "expand" : (aa - int(aa)) > 0.
    }

@add_to_class(Visualize_Linear_Data)
def plot_raw_points(self,fig):
    md = self.get_meta_data()
    n1 = md.get("fn")
    n2 = md.get("frame_size")
    n3 = md.get("expand")

    x_ = self.data_dict.get("features")
    y_ = self.data_dict.get("target")

    corr = self.correlations()
    axs = []

    for i in range(n1):
        ax = fig.add_subplot(n2 + (1 if n3 else 0),n2 + (1 if n3 else 0),i+1)
        axs.append(ax)
        ax.scatter(x_[f"x{i}"].to_numpy(), y_["y"],color = "red", alpha = 0.3)
        ax.set_title(f"correlation = {corr[i]}")
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel("y")
        ax.grid(True)
    return axs

@add_to_class(Visualize_Linear_Data)
def correlations(self):
    ans_tab = []
    for key in self.data_dict["features"].keys():
        x_d = self.data_dict["features"][key].to_numpy()
        y_d = self.data_dict["target"]["y"]
        std_ = ((x_d - np.mean(x_d))@(y_d - np.mean(y_d))) / (np.std(x_d) * np.std(y_d)) / len(x_d)
        ans_tab.append(np.round(std_,5))
    return ans_tab

########################################################################################


class LinearRegression_cf:
    def __init__(self,alpha = 0.01,steps = 1000):
        self.alpha = alpha
        self.steps = steps

    def fit(self,X,y,batch_size = 1,seed = 42):
        #default setting : Stochastic gradient descent
        raise NotImplementedError

    def batch_grad(self,X,y,B):
        raise NotImplementedError
    
    def Cost_function(self,X,y):
        raise NotImplementedError
    
    def PreProcessing(self,X,y):
        raise NotImplementedError
    
@add_to_class(LinearRegression_cf)
def fit(self,X,y,batch_size = 1,seed = 42,collect_data = False,scale_features = False):
    assert isinstance(X,np.ndarray) , "Invalid input datatype for X"
    assert isinstance(y,np.ndarray) , "Invalid input datatype for y"

    if len(X.shape) == 1:
        n = X.shape
        d = 1
    else:
        n,d = X.shape

    X_prime,y_prime,means_x,stds_x,means_y,stds_y = self.PreProcessing(X,y,scale_features)
    X_bias = np.hstack((X_prime,np.ones(n).reshape(-1,1)))
    rng = np.random.default_rng(seed)
    self.w_ = rng.normal(loc = 0. , scale = 0.1,size = d+1)
    L_data = []
    w_data = []

    for _ in range(self.steps):
        if collect_data:
            L_data.append(self.Cost_function(X_bias,y))
            w_data.append(self.w_)
        B = np.random.choice(n,size = batch_size)
        self.w_ += self.alpha / len(B) * self.batch_grad(X_bias,y_prime,B)
        
    self.rescale_features(means_x,stds_x,means_y,stds_y,scale_features)
    if collect_data:
        return self.w_,L_data,w_data
    else:
        return self.w_

@add_to_class(LinearRegression_cf)
def rescale_features(self,means_x,stds_x,means_y,stds_y,scale_features = True):
    if scale_features:
        self.w_[-1] = stds_y*self.w_[-1]+means_y - sum(means_x * self.w_[:-1] /stds_x) * stds_y
        self.w_[:-1] = self.w_[:-1] * stds_y / stds_x
    
@add_to_class(LinearRegression_cf)
def PreProcessing(self,X,y,scale_features = True):
    means_x = np.mean(X,axis = 0)
    stds_x = np.std(X,axis = 0)

    means_y = np.mean(y)
    stds_y = np.std(y)
    if scale_features:
        return (X-means_x)/stds_x, (y-means_y)/stds_y, means_x,stds_x,means_y,stds_y
    else:
        return X, y, means_x,stds_x,means_y,stds_y

@add_to_class(LinearRegression_cf)
def batch_grad(self,X,y,B):
    return (X[B]).T @ ((y - X @ self.w_)[B])

@add_to_class(LinearRegression_cf)
def Cost_function(self,X,y):
    side1 = (y - X @ self.w_)
    return side1.T @ side1 / X.shape[0]


########################################################################################

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

class SVC:
    def __init__(self):
        pass

    def fit(self,X_data,y_data):
        raise NotImplementedError
    
    def predict(self,X_data):
        raise NotImplementedError
    
    def phi(self,X_data):
        raise NotImplementedError
    
@add_to_class(SVC)
def fit(self, X_data, y_data,**kwargs):
    if len(X_data.shape) == 2:
        n, d = X_data.shape
    else:
        n = X_data.shape[0]
        d = 1
    
    classes = np.unique(y_data)
    y_data = np.where(y_data == classes[0], -1, 1)

    X_data_prime = self.phi(X_data)
    K = X_data_prime @ X_data_prime.T

    P = matrix(np.outer(y_data, y_data) * K) 
    q = matrix(-np.ones(n))                               
    A = matrix(y_data.astype('double'), (1, n)) 
    b = matrix(0.0)                       
    if "C" in kwargs.keys():
        C = kwargs.get("C")
        G = matrix(np.vstack((-np.eye(n),np.eye(n))))                   
        h = matrix(np.hstack((np.zeros(n),np.ones(n)*C))) 
    else:
        G = matrix(-np.eye(n))                   
        h = matrix(np.zeros(n)) 
    
    lam = np.ravel(solvers.qp(P, q, G, h, A, b)["x"])

    eps = lam > 1e-5 # support vectors

    self.w_ = np.sum(lam[:,np.newaxis] * X_data_prime * y_data[:,np.newaxis], axis=0)
    self.b_ = np.mean(y_data[eps] - (X_data_prime @ self.w_)[eps]) 

    return self.w_ , self.b_

@add_to_class(SVC)
def predict(self,X_data):
    return np.sign(self.phi(X_data) @ (self.w_) + self.b_)

    