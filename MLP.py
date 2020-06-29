from Library import NN_lib


class MLP(NN_lib):
    
    def __init__(self, layers, act_funcs):
        
        global np
        import numpy as np
        
        super().__init__()
        
        self.epsilon = 1e-5
        self.layers = layers
        self.act_funcs = act_funcs
        
        self.act_func_map = {'relu':self.ReLU, 'sigmoid':self.sigmoid, 'tanh':self.tanh, 
                             'none':self.none, 'softmax':self.softmax}
        
        self.act_diff_map = {'relu':self.ReLU_diff, 'sigmoid':self.sigmoid_diff, 
                             'tanh':self.tanh_diff, 'softmax':self.softmax_diff}
        
        self.loss_error_map = {'mse':self.mse_error, 'binary_crossentropy':self.bincross_error, 
                               'categorical_crossentropy':self.catcross_error, 
                               'sparse_categorical_crossentropy':self.sparcat_error}
        
        self.metric_map = {'mse':self.calc_mse, 'binary_crossentropy':self.calc_bincross, 
                           'categorical_crossentropy':self.calc_catcross, 
                           'accuracy':self.calc_accuracy, 
                           'sparse_categorical_crossentropy':self.calc_sparcat}
        
        self.weights = [ [] for _ in range(len(self.layers)) ]
        self.biases = [ [] for _ in range(len(self.layers)) ]  
        self.activations = [ [] for _ in range(len(self.layers)) ]
        
        for i in range(1, len(layers)):
            
            self.weights[i] = np.random.randn(layers[i-1], layers[i]) * 2/np.sqrt(layers[i])
            self.biases[i] = np.zeros(layers[i])
            
    
        
    def fit(self, X, y, **kwargs):  
        
        if len(y.shape) == 1:
            y = np.expand_dims(y, 1)
          
        self.epochs = kwargs.get('epochs', None)    
        self.eta = kwargs.get('learning_rate', None)    
        self.valid = kwargs.get('validation_data', None)  
        self.batch_size = kwargs.get('batch_size', len(X))  
        
        self.history = {}
        
        for metric in self.metrics:
            self.history[metric] = [] 
            if self.valid != None:
                self.history[f'val_{metric}'] = []   
                if len(self.valid[1].shape) == 1:
                    self.valid[1] = np.expand_dims(self.valid[1], 1)
            
        for epoch in range(1, self.epochs+1):
            
            n_batches = np.ceil(len(X)/self.batch_size).astype(int)       
            i = 0     
            
            for _ in range(n_batches):  
                
                if i+self.batch_size > len(X):        
                    batch_inp = X[i:]  
                    batch_targets = y[i:]
                else:  
                    batch_inp = X[i:i+self.batch_size]
                    batch_targets = y[i:i+self.batch_size]
                    
                self.forward_pass(batch_inp)
                self.back_propagation(batch_targets)
                i += self.batch_size
            
            for metric in self.metrics:
                self.history[metric].append(self.metric_map[metric](batch_inp, batch_targets))
                if self.valid != None:
                    self.history[f'val_{metric}'].append(self.metric_map[metric](self.valid[0], self.valid[1]))
            
            print(f'\nEpoch {epoch} ==> ', end='')
            for k,v in self.history.items():
                print(f'{k}: {v[-1]}', end=' ')
            
    def forward_pass(self, X):  
        self.activations[0] = X
        for i in range(1, len(self.layers)):
            act_func = self.act_func_map[self.act_funcs[i]]
            self.activations[i] = act_func(np.dot(self.activations[i-1], self.weights[i]) + self.biases[i])
            
    def back_propagation(self, y):
        
        y_hat = self.activations[-1]
        e = self.loss_func_error(y, y_hat)
        
        for i in range(len(self.weights)-1, 0, -1):
            
            if i<len(self.weights)-1:
                e = np.dot(e, self.weights[i+1].T)
                
            if self.act_funcs[i] != 'none' :
                act_diff = self.act_diff_map[self.act_funcs[i]](self.activations[i])
                e = np.multiply(e, act_diff)
            
            bias_gradients = self.clip_gradients(np.sum(e, axis=0))
            weight_gradients = self.clip_gradients(np.dot(self.activations[i-1].T, e))
            self.weights[i] = self.weights[i] - self.eta*weight_gradients
            self.biases[i] = self.biases[i] - self.eta*bias_gradients

