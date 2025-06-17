import tensorflow as tf
from tensorflow import keras
import sklearn
import numpy as np
import pandas as pd

print("TensorFlow版本:", tf.__version__)
print("Keras版本:", keras.__version__)
print("scikit-learn版本:", sklearn.__version__)
print("NumPy版本:", np.__version__)
print("Pandas版本:", pd.__version__)

# 测试GPU是否可用
print("\nGPU可用:", tf.config.list_physical_devices('GPU'))