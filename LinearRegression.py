class LinearRegression():
    
    def __init__(self, eta=0.1):
        
        global np
        import numpy as np
        
        self.eta = eta
    
    def fit(self, X, y):
        
        n_features = X.shape[1]
        n_samples = y.shape[0]
        self.weights = np.random.randn(n_features)
        self.bias = 0
        
        for _ in range(1000):
            
            y_hat = self.predict(X)
            errors = (y - y_hat)
            
            weight_gradients =  -(2/n_samples)*np.dot(X.T, errors)
            bias_gradient =  -(2/n_samples)*np.sum(errors)
            
            self.weights = self.weights - self.eta*weight_gradients
            self.bias = self.bias - self.eta*bias_gradient
            
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def evaluate(self, X, y):
        n_samples = y.shape[0]
        y_hat = self.predict(X)
        mse = np.sum((y_hat - y)**2)/n_samples
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum(np.square(y - y_hat))) / (np.sum(np.square(y - np.mean(y))))
        print(f"Root mean square error: {rmse}")
        print(f"R2 score: {r2}")

