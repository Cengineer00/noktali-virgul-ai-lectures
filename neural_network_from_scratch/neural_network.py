import numpy as np

class LinearLayer():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # He Initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, input_data):
        self.input_data = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input_data.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * np.sum(output_gradient, axis=0, keepdims=True)

        return input_gradient
    
class ReLULayer():
    def forward(self, input_data):
        self.input_data = input_data
        return np.maximum(0, input_data)

    def backward(self, output_gradient, learning_rate):
        input_gradient = output_gradient.copy()
        input_gradient[self.input_data <= 0] = 0
        return input_gradient
    
class SoftmaxLayer():
    def forward(self, input_data):
        exps = np.exp(input_data)
        self.output_data = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output_data

    def backward(self, output_gradient, learning_rate):
        return output_gradient  # Gradient is passed through for cross-entropy loss
    
class CrossEntropyLoss():
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        m = targets.shape[0]
        log_likelihood = -np.log(predictions[range(m), targets])
        loss = np.sum(log_likelihood) / m
        return loss

    def backward(self):
        # self.targets = [0, 1, 2, ..., 9] for each sample
        # self.predictions = softmax outputs, shape (m, 10)
        # Y^ - Y
        # self.predictions[0] = [0.1, 0.2, 0.7] for sample [0, 0, 1]
        # [0, 0, 1] -> [2]
        m = self.targets.shape[0]
        grad = self.predictions.copy()
        one_hot_targets = np.zeros_like(self.predictions)
        one_hot_targets[np.arange(m), self.targets] = 1
        grad -= one_hot_targets
        grad /= m
        return grad
        

class NeuralNetwork():
    def __init__(self, input_size=784, hidden_size=10, output_size=10):
        self.W1 = LinearLayer(input_size, hidden_size)
        self.activation1 = ReLULayer()
        self.W2 = LinearLayer(hidden_size, output_size)
        self.softmax = SoftmaxLayer()
        self.activation2 = ReLULayer()
        self.loss_function = CrossEntropyLoss()

    def forward(self, x):
        out = self.W1.forward(x)
        out = self.activation1.forward(out)
        out = self.W2.forward(out)
        out = self.activation2.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, loss_grad, learning_rate):
        grad = self.softmax.backward(loss_grad, learning_rate)
        grad = self.activation2.backward(grad, learning_rate)
        grad = self.W2.backward(grad, learning_rate)
        grad = self.activation1.backward(grad, learning_rate)
        grad = self.W1.backward(grad, learning_rate)