class NN_lib():
    
    def __init__(self):
        
        global np
        import numpy as np
        
        self.epsilon = 1e-5
        
        self.act_func_map = {'relu':self.ReLU, 'sigmoid':self.sigmoid, 'tanh':self.tanh, 
                             'none':self.none, 'softmax':self.softmax}
        
        self.act_diff_map = {'relu':self.ReLU_diff, 'sigmoid':self.sigmoid_diff, 
                             'tanh':self.tanh_diff, 'softmax':self.softmax_diff, 'none':self.none_diff}
        
        self.loss_error_map = {'mse':self.mse_error, 'binary_crossentropy':self.bincross_error, 
                               'categorical_crossentropy':self.catcross_error, 
                               'sparse_categorical_crossentropy':self.sparcat_error}
        
        self.metric_map = {'mse':self.calc_mse, 'binary_crossentropy':self.calc_bincross, 
                           'categorical_crossentropy':self.calc_catcross, 
                           'accuracy':self.calc_accuracy, 
                           'sparse_categorical_crossentropy':self.calc_sparcat}
        
    
    def tanh(self, X):
        return np.divide( (np.exp(X) - np.exp(-X)), (np.exp(X) + np.exp(-X) + self.epsilon) )
    
    def none(self, X):
        return X
    
    def sigmoid(self, X):
        f = np.vectorize(self.stable_sigmoid)
        return f(X)
        
    def stable_sigmoid(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    
    def ReLU(self, X):
        return np.maximum(X, 0)
    
    def softmax(self, X):
        X = X - np.max(X, axis=1, keepdims=True)
        numerator = np.exp(X)
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return np.divide(numerator, denominator)
    
    def tanh_diff(self, X):
        return 1 - np.square(self.tanh(X))
    
    def sigmoid_diff(self, X):
        return np.multiply(self.sigmoid(X), (1-self.sigmoid(X)))
    
    def ReLU_diff(self, X):
        X[X<=0] = 0
        X[X>0] = 1
        return X
    
    def none_diff(self, X):
        return np.ones(X.shape)
    
    def softmax_diff(self, X):
        return np.multiply(self.softmax(X), (1-self.softmax(X)))
    
        
    def compile_network(self, loss, *args):
        
        if len(args) == 1:
            self.metrics = args[0]
        else:
            self.metrics = []
        self.metrics.insert(0, loss)
        self.loss_func_error = self.loss_error_map[loss]
        self.loss = loss
        
    def predict(self, X, return_sequences=False):
        self.forward_pass(X)
        
        if hasattr(self, 'sequences'):
            results = self.sequences
        else:
            results = self.activations
        if self.loss == 'binary_crossentropy':
            if return_sequences==False:
                return (results[-1] > 0.5).astype(int)
            else:
                return (results > 0.5).astype(int)
        else:
            if return_sequences==False:
                return results[-1]
            else:
                return results
        
    def clip_gradients(self, grad):
        grad[grad>1] = 1
        grad[grad<-1] = -1
        return grad
        
    def mse_error(self, y, y_hat):      
        n_samples = y.shape[0]
        e = -(2/n_samples)*(y-y_hat)
        return e
            
    def bincross_error(self, y, y_hat):     
        n_samples = y.shape[0]
        e = (1/n_samples)*( -np.divide(y, y_hat+ self.epsilon) + np.divide((1-y), (1-y_hat+ self.epsilon)) )   
        return e
    
    def catcross_error(self, y, y_hat):     
        n_classes = y.shape[1]
        n_samples = y.shape[0]
        e = (1/(n_samples*n_classes))*( -np.divide(y, y_hat + self.epsilon) ) 
        self.kk = e  
        return e

    def sparcat_error(self, y, y_hat):
        n_samples = y_hat.shape[0]
        n_classes = y_hat.shape[1]
        sample_inds = np.arange(0, n_samples).reshape(-1, 1)
        e = np.zeros(y_hat.shape)
        e[sample_inds, y] = 1/(y_hat[sample_inds, y] + self.epsilon)
        e = -(1/n_samples*n_classes)*e
        self.kk = e
        return e
    
    def calc_mse(self, X, y):
        n_samples = y.shape[0]
        y_hat = self.predict(X)
        return np.sum(np.square(y - y_hat))/n_samples
    
    def calc_bincross(self, X, y):
        n_samples = y.shape[0]
        self.forward_pass(X)
        y_hat = self.activations[-1]
        return -np.sum(np.multiply(y, np.log(y_hat + self.epsilon)) + np.multiply((1-y), np.log(1-y_hat + self.epsilon)) ) / n_samples
    
    def calc_catcross(self, X, y):
        n_samples = y.shape[0]
        n_classes = y.shape[1]
        self.forward_pass(X)
        y_hat = self.activations[-1]
        return -np.sum(np.multiply(y, np.log(y_hat + self.epsilon)))/(n_samples*n_classes)
    
    def calc_sparcat(self, X, y):
        n_samples = y.shape[0]
        self.forward_pass(X)
        y_hat = self.activations[-1]
        n_classes = y_hat.shape[1]
        sample_inds = np.arange(0, n_samples).reshape(-1, 1)
        y_target_probs = y_hat[sample_inds, y]
        return -np.sum(np.log(y_target_probs + self.epsilon))/(n_samples*n_classes)

    def calc_rmse(self, X, y):
        return np.sqrt(self.calc_mse(X, y))    
        
    def calc_r2(self, X, y):
        n_samples = y.shape[0]
        y_hat = self.predict(X)
        r2 = 1 - (np.sum(np.square(y - y_hat))) / (np.sum(np.square(y - np.mean(y))))
        return r2
    
    def calc_accuracy(self, X, y):
        n_samples = y.shape[0]
        if self.loss == 'binary_crossentropy':
            y_hat = (self.predict(X) > 0.5).astype(int)
            return np.sum(y == y_hat) / n_samples
        elif self.loss == 'categorical_crossentropy':
            y_hat = np.argmax(self.predict(X), axis=1)
            y =  np.argmax(y, axis=1)
            return np.sum(y == y_hat) / n_samples
        elif self.loss == 'sparse_categorical_crossentropy':
            y = y.flatten()
            y_hat = np.argmax(self.predict(X), axis=1)
            return np.sum(y == y_hat) / n_samples

    def evaluate(self, X, y):
        
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)
        
        if self.loss == 'mse':
            print(f"Root mean square error: {self.calc_rmse(X, y)}")
            print(f"R2 score: {self.calc_r2(X, y)}")
            
        else:
            print(f"Accuracy: {self.calc_accuracy(X, y)}")

