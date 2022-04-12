# AIG Classifier

Experimental scheme to use AIGs as binary classifiers. Initially a random AIG is built. We therefore specify the number of layers and the number of nodes per layer. Each node takes two parents and if these inputs are negated depends on the random initialization. Using a local greedy search, we try to make the architecture perform better on the dataset (we are currently using binary MNIST). A picture of a random AIG is below. The nodes that are not active are dashed.

![random AIG](random_aig.jpg)

## Files

- `main.cpp`: Main program that performs the simulation.
- `prepro.ipynb`: Jupyter notebook where preprocessing is performed. The underlying functions are in `mnist.py`. The preprocessing done now is to read LeCun's MNIST files, perform PCA to reduce dimensionality and save the data in the same format as the original MNIST files.

## Installation

To compile the C++ code:

```
gcc main.cpp -lstdc++ -o main
```

To run the Python code, just make sure Numpy and Scikit-learn are installed.
