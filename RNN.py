from Library import NN_lib


class RNN(NN_lib):
    
    def __init__(self, inp_shape, hidden_nodes, out_nodes, act_func):
        
        global np
        import numpy as np
        
        super().__init__()
    
        self.time_steps = inp_shape[0]
        self.U = np.random.randn(inp_shape[1], hidden_nodes) * 2/np.sqrt(hidden_nodes)
        self.W = np.random.randn(hidden_nodes, hidden_nodes) * 2/np.sqrt(hidden_nodes)
        self.V = np.random.randn(hidden_nodes, out_nodes) * 2/np.sqrt(out_nodes)
        self.hidden_b = np.zeros(hidden_nodes)
        self.out_b = np.zeros(out_nodes)
        
        self.act_func = self.act_func_map[act_func]
        self.act_diff = self.act_diff_map[act_func]
        self.sequences = []
        self.states = []
    
            
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
                self.back_propagation(batch_inp, batch_targets)
                i += self.batch_size
            
            for metric in self.metrics:
                self.history[metric].append(self.metric_map[metric](X, y))
                if self.valid != None:
                    self.history[f'val_{metric}'].append(self.metric_map[metric](self.valid[0], self.valid[1]))
            
            print(f'\nEpoch {epoch} ==> ', end='')
            for k,v in self.history.items():
                print(f'{k}: {v[-1]}', end=' ')
                
    def forward_pass(self, X):
        self.sequences = []
        self.states = []
        self.states.append(np.zeros((X.shape[0], self.U.shape[1])))
        for time_step in range(self.time_steps):
            self.states.append(self.tanh(np.dot(X[:, time_step, :], self.U) + np.dot(self.states[-1], self.W) + self.hidden_b))
            self.sequences.append(self.act_func(np.dot(self.states[-1], self.V) + self.out_b))
            
    def back_propagation(self, X, y):
        
        y_hat = self.sequences[-1]
        e = self.loss_func_error(y, y_hat)
        
        e = np.multiply(e, self.act_diff(y_hat))
            
        grad_out_b = self.clip_gradients(np.sum(e, axis=0))
        
        grad_V = self.clip_gradients(np.dot(self.states[-1].T, e))
        
        foo = np.multiply(np.dot(e, self.V.T), self.states[-1])
        
        bar_w = self.states[0]
        
        for i in range(1, len(self.states)-1):
            
            bar_w = np.dot(np.multiply(bar_w, self.tanh_diff(self.states[i])), self.W.T) + self.states[i]
            
        grad_W = self.clip_gradients(np.dot(bar_w.T, foo))
        
        bar_u = np.dot(X[:, self.time_steps-1, :].T, foo)
        
        bar_hb = foo
        
        dummy_u = np.identity(self.W.shape[0])
        
        for i in range(self.time_steps-2, -1, -1):
                
            dummy_u = np.multiply(dummy_u, self.W.T)
            
            foo = np.multiply(foo, self.act_diff(self.states[i+1]))
            
            bar_u += np.dot(X[:, i, :].T, np.dot(foo, dummy_u))
                            
            bar_hb += np.dot(bar_hb, dummy_u)
                            
        grad_U = self.clip_gradients(bar_u)
                            
        grad_hidden_b = self.clip_gradients(np.sum(bar_hb, axis=0))
            
        self.U = self.U - self.eta*grad_U
        self.W = self.W - self.eta*grad_W
        self.V = self.V - self.eta*grad_V
        self.hidden_b = self.hidden_b - self.eta*grad_hidden_b
        self.out_b = self.out_b - self.eta*grad_out_b
            
    def view_hidden_state(self, view_all=False):
        if view_all:
            return self.states
        else:
            return self.states[-1]
