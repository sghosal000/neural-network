import numpy as np

class MSE:
  def forward(self, y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
  
class CategoricalCrossEntropy:
  def forward(self, y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1-1e-7)
    retur