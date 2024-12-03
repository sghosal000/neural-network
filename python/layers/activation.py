import numpy as np

class ReLU:
  def forward(self, inputs):
    return np.maximum(0, inputs)
  
class SoftMax:
  def forward(self, inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return self.output
  
class Sigmoid:
  def forward(self, inputs):
    self.output = 1 / (1 + np.exp(-inputs))
    return self.output
  
class Tanh:
  def forward(self, inputs):
    # self.output = np.tanh(inputs)
    self.output = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
    return self.output
  
class LeakyReLU:
  def __init__(self, alpha):
    self.alpha = alpha

  def forward(self, inputs):
    self.output = np.where(inputs > 0, inputs, self.alpha*inputs)
    return self.output
  
class ELU:
  def __init__(self, alpha):
    self.alpha = alpha

  def forward(self, inputs):
    self.output = np.where(inputs > 0, inputs, self.alpha*(np.exp(inputs) - 1))
    return self.output