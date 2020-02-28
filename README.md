# MyMLP
MyMLP is a multi-layer perceptron with mini-batch gradient descent and backpropagation algorithm. It used the  logistic sigmoid f(x) = 1 / (1 + exp(-x)) as the activation function. 

Installation
------
### Dependecies:
- python (Tested on python3)
- numpy

Currently, pip package is not available. To install this library, you should clone this project manually and add the entire files into your project.
```
git clone https://github.com/Frankuy/ANN-MLP.git
```
Usage
------
The current version only supports one hidden layer with customized number of units. Here is the example using this library with iris dataset.
```python
from sklearn.datasets import load_iris
import numpy as np
from MyMLP import MyMLP

iris = load_iris()

features = iris.data
target = iris.target

# Get numbers of class attribute
number_unique_output = len(np.unique(target))

weight = MyMLP(features,
               target, 
               learning_rate=0.1,
               mini_data=3,
               hidden_layer_unit=100,
               output_layer_unit=number_unique_output,
               max_epoch=1000)
print(weight)
```