class LogisticRegression():
    
    def __init__(self, eta=0.1, prediction_threshold=0.5):
        
        global np
        import numpy as np
        
        self.eta = eta
        self.threshold = prediction_threshold
        
    def fit(self, X, y):
        
        n_features = X.shape[1]
        n_samples = y.shape[0]
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        for _ in range(1000):
            
            y_hat = self.probability(X)
            errors = (y - y_hat)
            
            weight_gradients =  -np.dot(X.T, errors) / n_samples
            bias_gradient =  -np.sum(errors) / n_samples
            
            self.weights = self.weights - self.eta*weight_gradients
            self.bias = self.bias - self.eta*bias_gradient
            
    def probability(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def predict(self, X):
        return (self.probability(X) > self.threshold).astype(int)
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def evaluate(self, X, y):
        n_samples = y.shape[0]
        y_hat_num = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_hat_bin = (y_hat_num > self.threshold).astype(int)
        binary_crossentropy_loss = (np.dot(y, np.log(y_hat_num)) + np.dot((1-y), np.log(1-y_hat_num)) ) / n_samples
        accuracy = np.sum(y == y_hat_bin) / n_samples
        print(f"Binary Cross-entropy loss: {binary_crossentropy_loss}")
        print(f"Accuracy: {accuracy}")

