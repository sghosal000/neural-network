import numpy as np

class Dense:
  def __init__(self, input_size, neurons, seed=0):
    np.random.seed(seed)
    self.weights = np.random.randn(input_size, neurons)*0.01
    self.biases = np.zeros(neurons)

  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(inputs, self.weights) + self.biases
    return self.output
