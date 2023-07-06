#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>
#include <experimental/algorithm>
#include <regex>
#include <assert.h>
#include <dirent.h>
#include <sys/stat.h>
#include "omp.h"
#include <chrono>

#include "mnist.cpp"
#include "aig.cpp"
#include "ml.cpp"

int main (int argc, char *argv[]) {
  //std::vector<int> num_nodes{5, 4, 4, 1};
  //std::vector<int> num_nodes{8, 16, 16, 16, 16, 16, 16, 16, 16, 1};
  //std::vector<int> num_nodes{8, 128, 64, 32, 16, 8, 1};
  std::vector<int> num_nodes{8, 512, 512, 512, 128, 64, 32, 16, 8, 1};
  while (true) {
  NodeNetwork nn;
  InitializeNodeNetwork(nn, num_nodes);
  // hello

  // Getting all active nodes
  std::vector<Node*> active_nodes;
  GetActiveNodes(nn.nodes.back()[0], active_nodes);
  GetUniquesAndSort(active_nodes);

  //ExportDot(nn, "test.dot");
  //ExportAigToPdf(nn, "test.pdf");

  //// Reading train images and binarizing
  //std::vector<std::vector<double>> X_train_;
  //ReadMNIST(60000,784,"data/train-images.idx3-ubyte", X_train_);
  //std::vector<std::vector<bool>> X_train;
  //BinarizeMNIST(X_train_, X_train);

    std::ifstream file("iwls2022-ls-contest/benchmarks/ex02.truth");
    if (!file) {
        std::cerr << "Unable to open file\n";
        return 1;   // call system to stop
    }

    std::string line;
    getline(file, line);
    file.close();

    int arity = std::log2(line.length());

    std::vector<std::vector<bool>> X_train = generateCombinations(arity);

    std::vector<bool> y_train;
    for (char c : line) {
        y_train.push_back(c - '0');
    }

    std::vector<std::vector<bool>> X_test = X_train;
    std::vector<bool> y_test = y_train;

  //// Reading PCA-reduced train images and binarizing
  //std::vector<std::vector<double>> X_train_;
  //ReadMNIST(60000,16,"data/train-images-pca16-idx3-ubyte", X_train_);
  //std::vector<std::vector<bool>> X_train;
  //X_train.resize(60000);
  //for (int i = 0; i < X_train_.size(); i++) {
    //for (int j = 0; j < X_train_[i].size(); j++) {
      //if (X_train_[i][j] == 1.0) {
        //X_train[i].push_back(true);
      //} else {
        //X_train[i].push_back(false);
      //}
    //}
  //}

  assert(num_nodes[0] == X_train[0].size()); // input dimension has to match
  assert (num_nodes.back() == 1); // for now we only do binary classification

  //// Reading train labels and binarizing
  //std::vector<double> y_train_;
  //ReadMNISTLabels(60000, "data/train-labels.idx1-ubyte", y_train_);
  //std::vector<bool> y_train;
  //BinarizeMNISTLabels(y_train_, y_train);

  // Less computation time
  //X_train.resize(10000);
  //y_train.resize(10000);

  // Reading test images and binarizing
  //std::vector<std::vector<double>> X_test_;
  //ReadMNIST(10000,784,"data/t10k-images.idx3-ubyte", X_test_);
  //std::vector<std::vector<bool>> X_test;
  //BinarizeMNIST(X_test_, X_test);

  // Reading test labels and binarizing
  //std::vector<double> y_test_;
  //ReadMNISTLabels(10000, "data/t10k-labels.idx1-ubyte", y_test_);
  //std::vector<bool> y_test;
  //BinarizeMNISTLabels(y_test_, y_test);

  /*
   Change nothing:    0
   Change polarity 0: 1
   Change polarity 1: 2
   Change parent 0:   3
   Change parent 1:   4
   Vector: nodes
   Vector: change {0, 1, 2, 3, 4}
   Vector: parent
   Vector: accuracies
  */

  // Those will be overwritten frequently
  std::vector<bool> pred;
  std::vector<bool> test_pred;
  double score;
  double test_score;
  Node* old_parent;
  int no_change = 0;
  std::vector<double> accuracies;
  std::vector<int> mode;
  std::vector<Node*> node_candidates;
  std::vector<Node*> parent_candidates;
  double best = 0;
  int iteration = 0;
  std::string out_str = "";
  std::string out_str_tmp = "";
  // End

  // These are parameters
  int patience = 5;
  double tol = 0.001;

  out_str_tmp += "arc ";
  for (auto layer : nn.nodes) {
    out_str_tmp += std::to_string(layer.size()) + " ";
  }
  out_str_tmp += "\n";
  std::cout << out_str_tmp;
  out_str += out_str_tmp;
  out_str_tmp = "";

  while (no_change < patience) {
  pred = Predict(nn, X_train, "tmp");
  score = AccuracyScore(y_train, pred);
  accuracies.push_back(score);
  mode.push_back(0); // 0 stands for "change nothing"
  node_candidates.emplace_back(); // Adding empty item
  parent_candidates.emplace_back(); // Adding empty item

  //auto start = std::chrono::high_resolution_clock::now();

  for (Node* node : active_nodes) {
    SearchAroundNode(
      node,
      nn,
      pred,
      score,
      old_parent,
      accuracies,
      mode,
      node_candidates,
      parent_candidates,
      X_train,
      y_train);
  }

    //auto stop = std::chrono::high_resolution_clock::now();
    //auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    //std::cout << "Iter took " << duration.count() << " seconds" << std::endl;

  int max_index = std::max_element(accuracies.begin(), accuracies.end()) - accuracies.begin();
  double max = *std::max_element(accuracies.begin(), accuracies.end());
  //



  // Get the top 5 accuracies
  //std::vector<double> top5(5);
  //std::vector<double> acc_cpy = accuracies;
  //std::partial_sort_copy(acc_cpy.begin(), acc_cpy.end(), top5.begin(), top5.end(), std::greater<double>());

  //// Subtract the minimum accuracy from each of the top 5
  ////double minAccuracy = *std::min_element(top5.begin(), top5.end());
  ////std::transform(top5.begin(), top5.end(), top5.begin(), [minAccuracy](double acc) { return acc - minAccuracy; });

  //// Normalize the top 5 accuracies
  //std::vector<double> top5_cpy = top5;
  //double sum = std::accumulate(top5_cpy.begin(), top5_cpy.end(), 0.0);
  //std::transform(top5_cpy.begin(), top5_cpy.end(), top5_cpy.begin(), [sum](double acc) { return acc / sum; });
  //for (auto x : top5_cpy) {
    //std::cout << x << " ";
  //}
  //std::cout << std::endl;

  //// Create a random number generator
  //std::random_device rd;
  //std::mt19937 gen(rd());
  //std::discrete_distribution<> dist(top5_cpy.begin(), top5_cpy.end());

  //// Sample from the top 5 accuracies
  //int sampleIndex = dist(gen);
  //double sampledAccuracy = top5[sampleIndex];

  //// Find the original index of the sampled accuracy in the 'accuracies' vector
  //std::vector<double>::iterator it = std::find(accuracies.begin(), accuracies.end(), sampledAccuracy);
  //int max_index = std::distance(accuracies.begin(), it);
  //double max = accuracies[max_index];




  if (max > best + tol) {
    best = max;
    no_change = 0;
    switch (mode[max_index]) {
      case 0:
        break; // do nothing
      case 1:
        node_candidates[max_index]->negates[0] = !node_candidates[max_index]->negates[0];
        break;
      case 2:
        node_candidates[max_index]->negates[1] = !node_candidates[max_index]->negates[1];
        break;
      case 3:
        node_candidates[max_index]->parents[0]->children.erase(std::find(node_candidates[max_index]->parents[0]->children.begin(),node_candidates[max_index]->parents[0]->children.end(),node_candidates[max_index]));
        node_candidates[max_index]->parents[0] = parent_candidates[max_index];
        node_candidates[max_index]->parents[0]->children.push_back(node_candidates[max_index]);
        break;
      case 4:
        node_candidates[max_index]->parents[1]->children.erase(std::find(node_candidates[max_index]->parents[1]->children.begin(),node_candidates[max_index]->parents[1]->children.end(),node_candidates[max_index]));
        node_candidates[max_index]->parents[1] = parent_candidates[max_index];
        node_candidates[max_index]->parents[1]->children.push_back(node_candidates[max_index]);
        break;
    }
  } else {
    no_change += 1;
  }

  test_pred = Predict(nn, X_test, "tmp");
  test_score = AccuracyScore(y_test, test_pred);

  out_str_tmp = "Iter " + std::to_string(iteration) + " Training accuracy " + std::to_string(max) + " Testing accuracy " + std::to_string(test_score) + "\n";
  std::cout << out_str_tmp;
  out_str += out_str_tmp;
  out_str_tmp = "";

  // Vectors have to be empty for next iteration
  accuracies.clear();
  mode.clear();
  node_candidates.clear();
  parent_candidates.clear();
  iteration += 1;

  // Active nodes may have changed after changing architecture
  active_nodes.clear();
  GetActiveNodes(nn.nodes.back()[0], active_nodes);
  GetUniquesAndSort(active_nodes);
  }

  SaveRun(nn, out_str, "results");
  }

  return 0;
}











