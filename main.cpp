#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <experimental/algorithm>
#include <regex>

#include "mnist.cpp"
#include "aig.cpp"
#include "ml.cpp"
// TODO Proper function and variable names

int main (int argc, char *argv[]) {
  //std::vector<int> num_nodes{5, 3, 3, 3, 1};
  std::vector<int> num_nodes{784, 512, 128, 32, 1};
  NodeNetwork nn;
  InitializeNodeNetwork(nn, num_nodes);

  //// Reading train images and binarizing
  std::vector<std::vector<double>> X_train_;
  ReadMNIST(60000,784,"data/train-images.idx3-ubyte", X_train_);
  std::vector<std::vector<bool>> X_train;
  BinarizeMNIST(X_train_, X_train);

  //// Reading train labels and binarizing
  std::vector<double> y_train_;
  ReadMNISTLabels(60000, "data/train-labels.idx1-ubyte", y_train_);
  std::vector<bool> y_train;
  BinarizeMNISTLabels(y_train_, y_train);

  //// Reading test images and binarizing
  //std::vector<std::vector<double>> X_test_;
  //ReadMNIST(10000,784,"data/t10k-images.idx3-ubyte", X_test_);
  //std::vector<std::vector<bool>> X_test;
  //BinarizeMNIST(X_test_, X_test);

  //// Reading test labels and binarizing
  //std::vector<double> y_test_;
  //ReadMNISTLabels(10000, "data/t10k-labels.idx1-ubyte", y_test_);
  //std::vector<bool> y_test;
  //BinarizeMNISTLabels(y_test_, y_test);

  std::vector<std::vector<bool>> v;
  v.push_back({true, true, false, true, true});
  v.push_back({false, true, true, true, false});
  v.push_back({false, true, true, true, true});
  v.push_back({true, true, true, true, true});
  std::vector<bool> pred = Predict(nn, X_train, "tmp");
  std::cout << "Accuracy: " << AccuracyScore(y_train, pred) << std::endl;
  return 0;
}











