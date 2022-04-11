#pragma once

class Node {
  public:
    int type; // 0: input node 1: and node 2: output node
    int var_id;
    int loc0;
    int loc1;
    std::vector<Node*> parents;
    std::vector<Node*> children;
    std::vector<bool> negates;
    Node(int a, int b, int c, int d);
};

class NodeNetwork {
  public:
    int max_var_idx;
    std::vector<std::vector<Node>> nodes;
    NodeNetwork();
};

void PrintNNTypesLocs(NodeNetwork &nn);

void CreateAagRepr(NodeNetwork &nn);

void ExportAagRepr(NodeNetwork &nn, std::string export_path);

void CreateSimulationFile(std::vector<std::vector<bool>> &X, std::string export_path);

void GetActiveNodes(Node &node, std::vector<Node*> &locs);

void ExportAigToPdf(NodeNetwork &nn, std::string pdf_file);

std::vector<bool> Predict(NodeNetwork &nn, std::vector<std::vector<bool>> &X, std::string export_folder, bool keep_files = false);

void InitializeNodeNetwork(NodeNetwork &nn, std::vector<int> &num_nodes);

void ChangeParent(Node* node, int parent_idx, NodeNetwork &nn);

void SaveRun(NodeNetwork &nn, std::string save_folder);

void SearchAroundNode(
  Node* node,
  std::vector<bool> &pred,
  double &score,
  Node* old_parent,
  std::vector<double> &accuracies,
  std::vector<int> &mode,
  std::vector<Node*> &node_candidates,
  std::vector<Node*> &parent_candidates,
  std::vector<std::vector<bool>> &X_train,
  std::vector<bool> &y_train);
