#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <experimental/algorithm>
#include <regex>
#include <assert.h>

#include "mnist.cpp"
#include "aig.cpp"
#include "ml.cpp"

int main (int argc, char *argv[]) {
  //std::vector<int> num_nodes{5, 3, 3, 3, 1};
  std::vector<int> num_nodes{16, 16, 8, 4, 1};
  //std::vector<int> num_nodes{784, 512, 128, 32, 1};
  NodeNetwork nn;
  InitializeNodeNetwork(nn, num_nodes);

  // Getting all active nodes
  std::vector<Node*> active_nodes;
  GetActiveNodes(nn.nodes.back()[0], active_nodes);
  GetUniquesAndSort(active_nodes);

  //// Reading train images and binarizing
  //std::vector<std::vector<double>> X_train_;
  //ReadMNIST(60000,784,"data/train-images.idx3-ubyte", X_train_);
  //std::vector<std::vector<bool>> X_train;
  //BinarizeMNIST(X_train_, X_train);

  //// Reading PCA-reduced train images and binarizing
  std::vector<std::vector<double>> X_train_;
  ReadMNIST(60000,16,"data/train-images-pca16-idx3-ubyte", X_train_);
  std::vector<std::vector<bool>> X_train;
  X_train.resize(60000);
  for (int i = 0; i < X_train_.size(); i++) {
    for (int j = 0; j < X_train_[i].size(); j++) {
      if (X_train_[i][j] == 1.0) {
        X_train[i].push_back(true);
      } else {
        X_train[i].push_back(false);
      }
    }
  }

  assert(num_nodes[0] == X_train[0].size()); // input dimension has to match
  assert (num_nodes.back() == 1); // for now we only do binary classification

  //// Reading train labels and binarizing
  std::vector<double> y_train_;
  ReadMNISTLabels(60000, "data/train-labels.idx1-ubyte", y_train_);
  std::vector<bool> y_train;
  BinarizeMNISTLabels(y_train_, y_train);

  // Less computation time
  X_train.resize(10000);
  y_train.resize(10000);

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

  //std::vector<std::vector<bool>> v;
  //v.push_back({true, true, false, true, true});
  //v.push_back({false, true, true, true, false});
  //v.push_back({false, true, true, true, true});
  //v.push_back({true, true, true, true, true});

  //std::vector<bool> pred = Predict(nn, X_train, "tmp");
  //std::cout << "Accuracy orig: " << AccuracyScore(y_train, pred) << std::endl;

  // Change nothing:    0
  // Change polarity 0: 1
  // Change polarity 1: 2
  // Change parent 0:   3
  // Change parent 1:   4
  // Vector: nodes
  // Vector: change {0, 1, 2, 3, 4}
  // Vector: parent
  // Vector: accuracies

  // Those will be overwritten frequently
  std::vector<bool> pred;
  double score;
  Node* old_parent;
  // End

  std::vector<double> accuracies;
  std::vector<int> mode;
  std::vector<Node*> node_candidates;
  std::vector<Node*> parent_candidates;

  pred = Predict(nn, X_train, "tmp");
  score = AccuracyScore(y_train, pred);
  accuracies.push_back(score);
  mode.push_back(0); // 0 stands for "change nothing"
  node_candidates.emplace_back(); // Adding empty item
  parent_candidates.emplace_back(); // Adding empty item

  for (Node* node : active_nodes) {
    // 1: Change polarity 0
    node->negates[0] = !node->negates[0];
    pred = Predict(nn, X_train, "tmp");
    score = AccuracyScore(y_train, pred);
    accuracies.push_back(score);
    mode.push_back(1); // 1 stands for "change polarity 0"
    node_candidates.push_back(node);
    parent_candidates.emplace_back(); // Adding empty item
    node->negates[0] = !node->negates[0]; // Reverting modes

    // 2: Change polarity 1
    node->negates[1] = !node->negates[1];
    pred = Predict(nn, X_train, "tmp");
    score = AccuracyScore(y_train, pred);
    accuracies.push_back(score);
    mode.push_back(2); // 2 stands for "change polarity 1"
    node_candidates.push_back(node);
    parent_candidates.emplace_back(); // Adding empty item
    node->negates[1] = !node->negates[1]; // Reverting modes

    // 3: Change parent 0
    old_parent = node->parents[0];
    ChangeParent(node, 0, nn);
    pred = Predict(nn, X_train, "tmp");
    score = AccuracyScore(y_train, pred);
    accuracies.push_back(score);
    mode.push_back(3); // 3 stands for "change parent 0"
    node_candidates.push_back(node);
    parent_candidates.push_back(node->parents[0]);
    // Deregistering node from new parent
    node->parents[0]->children.erase(std::find(node->parents[0]->children.begin(),node->parents[0]->children.end(),node));
    // Setting parent to old parent
    node->parents[0] = old_parent;
    // Registering node at old parent again
    node->parents[0]->children.push_back(node);

    // 4: Change parent 1
    old_parent = node->parents[1];
    ChangeParent(node, 1, nn);
    pred = Predict(nn, X_train, "tmp");
    score = AccuracyScore(y_train, pred);
    accuracies.push_back(score);
    mode.push_back(4); // 4 stands for "change parent 1"
    node_candidates.push_back(node);
    parent_candidates.push_back(node->parents[1]);
    // Deregistering node from new parent
    node->parents[1]->children.erase(std::find(node->parents[1]->children.begin(),node->parents[1]->children.end(),node));
    // Setting parent to old parent
    node->parents[1] = old_parent;
    // Registering node at old parent again
    node->parents[1]->children.push_back(node);
  }
  for (int i = 0; i < accuracies.size(); i++) {
    std::cout << "mode: " << mode[i] << " acc: " << accuracies[i] << std::endl;
  }

  return 0;
}











