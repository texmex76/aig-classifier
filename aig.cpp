#include "aig.hpp"
#include "utils.cpp"
#include "ml.hpp"

Node::Node (int a, int b, int c, int d) {type = a; var_id = b; loc0 = c; loc1 = d;}

NodeNetwork::NodeNetwork()
 :max_var_idx(0)
{}

void PrintNNTypesLocs(NodeNetwork &nn) {
  for (int i = 0; i < nn.nodes.size(); i++) {
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      std::string type_str;
      switch (nn.nodes[i][j].type) {
        case 0:
          type_str = "Input Node";
          break;
        case 1:
          type_str = "And Node";
          break;
        case 2:
          type_str = "Output Node";
          break;
        default:
          type_str = "Unknown Type";
          break;
      }
      std::cout << type_str << " at " << nn.nodes[i][j].loc0 << " " << nn.nodes[i][j].loc1 << std::endl;
      std::cout << "Parents:";
      for (int k = 0; k < nn.nodes[i][j].parents.size(); k++) {
        std::string neg = nn.nodes[i][j].negates[k] ? "~" : "";
        std::cout << neg << "(" << nn.nodes[i][j].parents[k]->loc0 << " " << nn.nodes[i][j].parents[k]->loc1 << ") ";
      }
      std::cout << "\n";
      std::cout << "Children:";
      for (Node* addr : nn.nodes[i][j].children) {std::cout << "(" << addr->loc0 << " " << addr->loc1 << ") ";}
      std::cout << "\n";
    }
  }
}

void CreateAagRepr(NodeNetwork &nn) {
  int num_and_gates = 0;
  for (int i = 1; i < nn.nodes.size(); i++) {num_and_gates += nn.nodes[i].size();}
  std::cout << "aag " << nn.max_var_idx << " " << nn.nodes[0].size() << " 0 " << nn.nodes.back().size() << " " << num_and_gates << std::endl;

  // Input nodes
  for (int j = 0; j < nn.nodes[0].size(); j++) {
    std::cout << nn.nodes[0][j].var_id * 2 << std::endl;
  }

  // Output nodes
  for (int j = 0; j < nn.nodes.back().size(); j++) {
    std::cout << nn.nodes.back()[j].var_id * 2 << std::endl;
  }

  // And nodes
  for (int i = 1; i < nn.nodes.size(); i++) {
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      std::cout << nn.nodes[i][j].var_id * 2 << " ";
      std::cout << nn.nodes[i][j].parents[0]->var_id * 2 + nn.nodes[i][j].negates[0] << " ";
      std::cout << nn.nodes[i][j].parents[1]->var_id * 2 + nn.nodes[i][j].negates[1] << std::endl;
    }
  }
}

void ExportAagRepr(NodeNetwork &nn, std::string export_path) {
  std::ofstream o;
  o.open(export_path);
  int num_and_gates = 0;
  for (int i = 1; i < nn.nodes.size(); i++) {num_and_gates += nn.nodes[i].size();}
  o << "aag " << nn.max_var_idx << " " << nn.nodes[0].size() << " 0 " << nn.nodes.back().size() << " " << num_and_gates << std::endl;

  // Input nodes
  for (int j = 0; j < nn.nodes[0].size(); j++) {
    o << nn.nodes[0][j].var_id * 2 << std::endl;
  }

  // Output nodes
  for (int j = 0; j < nn.nodes.back().size(); j++) {
    o << nn.nodes.back()[j].var_id * 2 << std::endl;
  }

  // And nodes
  for (int i = 1; i < nn.nodes.size(); i++) {
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      o << nn.nodes[i][j].var_id * 2 << " ";
      o << nn.nodes[i][j].parents[0]->var_id * 2 + nn.nodes[i][j].negates[0] << " ";
      o << nn.nodes[i][j].parents[1]->var_id * 2 + nn.nodes[i][j].negates[1] << std::endl;
    }
  }
  o.close();
}

void CreateSimulationFile(std::vector<std::vector<bool>> &X, std::string export_path) {
  std::ofstream o;
  o.open(export_path);
  for (int i = 0; i < X.size(); i++) {
    for (int j = 0; j < X[i].size(); j++) {
      o << X[i][j];
    }
    o << std::endl;
  }
  o << ".";
  o.close();
}

void ExportAigToPdf(NodeNetwork &nn, std::string pdf_file) {
  std::string aag_file = "tmp.aag_" + GenerateHex(10);
  ExportAagRepr(nn, aag_file);
  std::string dot_file = "tmp.dot_" + GenerateHex(10);

  std::string exec_aig_str = "aiger/aigtodot " + aag_file + " > " + dot_file;
  std::string exec_dot_str = "dot -Tpdf " + dot_file + " -o " + pdf_file;
  Exec(exec_aig_str.c_str());
  Exec(exec_dot_str.c_str());

  // Delete files after use
  const int remove_aag_file_result = remove(aag_file.c_str());
  const int remove_dot_file_result = remove(dot_file.c_str());
}

std::vector<bool> Predict(NodeNetwork &nn, std::vector<std::vector<bool>> &X, std::string export_folder, bool keep_files) {
  std::string stim_file = export_folder + "/stim_" + GenerateHex(10);
  CreateSimulationFile(X, stim_file);
  std::string aag_file = export_folder + "/tmp.aag_" + GenerateHex(10);
  ExportAagRepr(nn, aag_file);
  std::string exec_str = "aiger/aigsim -m " + aag_file + " " + stim_file;
  std::string out = Exec(exec_str.c_str());

  // Delete files after use
  if (!keep_files) {
    const int remove_stim_file_result = remove(stim_file.c_str());
    const int remove_aag_file_result = remove(aag_file.c_str());
  }

  const std::regex re("\\s(\\d+)\\s");
  while (std::regex_search(out, re)) {
      out = std::regex_replace(out, re, "hello");
  }

  const std::regex re2("hello");
  while (std::regex_search(out, re2)) {
      out = std::regex_replace(out, re2, "");
  }

  const std::regex re3("\n");
  while (std::regex_search(out, re3)) {
      out = std::regex_replace(out, re3, "");
  }

  const std::regex re4("\\sTrace is a.+");
  while (std::regex_search(out, re4)) {
      out = std::regex_replace(out, re4, "");
  }

  std::stringstream ss(out);
  std::istream_iterator<std::string> begin(ss);
  std::istream_iterator<std::string> end;
  std::vector<std::string> tokens(begin, end);

  std::vector<bool> pred;
  std::istringstream iss (out);
  for (auto &out: tokens) {
    bool val;
    iss >> val;
    pred.push_back(val);
  }

  return pred;
}

void InitializeNodeNetwork(NodeNetwork &nn, std::vector<int> &num_nodes) {
  std::vector<Node> inp_vec;
  nn.nodes.push_back(inp_vec);
  for (int j = 0; j < num_nodes[0]; j++) {
    nn.max_var_idx += 1;
    nn.nodes[0].push_back(Node(0, nn.max_var_idx, 0, j));
  }
  // And nodes
  for (int i = 1; i < num_nodes.size() - 1; i++) {
    // i is the row index
    // We start at 1 since the 0th row is the input row
    std::vector<Node> and_vec;
    nn.nodes.push_back(and_vec);
    for (int j = 0; j < num_nodes[i]; j++) {
      nn.max_var_idx += 1;
      nn.nodes[i].push_back(Node(1, nn.max_var_idx, i, j));
    }
  }
  // Output nodes
  std::vector<Node> out_vec;
  nn.nodes.push_back(out_vec);
  for (int j = 0; j < num_nodes.back(); j++) {
    nn.max_var_idx += 1;
    nn.nodes.back().push_back(Node(2, nn.max_var_idx, num_nodes.size() - 1, j));
  }
  // Connections
  for (int i = nn.nodes.size() - 1; i > 0; i--) {
    std::vector<int> col_idxs(nn.nodes[i - 1].size());
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      std::iota(col_idxs.begin(), col_idxs.end(), 0);
      std::vector<int> choice;
      std::experimental::sample(col_idxs.begin(), col_idxs.end(), std::back_inserter(choice),
                  2, std::mt19937{std::random_device{}()});
      for (int col_idx : choice) {
        nn.nodes[i][j].parents.push_back(&nn.nodes[i - 1][col_idx]);
        nn.nodes[i - 1][col_idx].children.push_back(&nn.nodes[i][j]);
        nn.nodes[i][j].negates.push_back(rand() % 2);
        nn.nodes[i][j].negates.push_back(rand() % 2);
      }
    }
  }
}

// Only works with 1 output node
void GetActiveNodes(Node &node, std::vector<Node*> &locs) {
  //std::cout << "(" << node.loc0 << ", " << node.loc1 << ") par: " << node.parents.size() << std::endl;
  if (node.loc0 == 0) {
    return;
  }
  locs.push_back(&node);
  for (Node* parent : node.parents) {
    GetActiveNodes(*parent, locs);
  }
}

//void ChangeParent(Node* node, int parent_idx, NodeNetwork &nn) {
  //assert(nn.nodes[node->loc0 - 1].size() > 2); // If the previous layer has only 2 nodes, then there is nothing to change
  //while (true) {
    //std::vector<int> col_idxs(nn.nodes[node->loc0 - 1].size());
    //std::iota(col_idxs.begin(), col_idxs.end(), 0);
    //std::vector<int> choice;
    //std::experimental::sample(col_idxs.begin(), col_idxs.end(), std::back_inserter(choice), 1, std::mt19937{std::random_device{}()});
    //if (choice[0] != node->parents[parent_idx]->loc1 && choice[0] != node->parents[!parent_idx]->loc1) {
      //node->parents[parent_idx]->children.erase(std::find(node->parents[parent_idx]->children.begin(),node->parents[parent_idx]->children.end(),node));
      //node->parents[parent_idx] = &nn.nodes[node->loc0 - 1][choice[0]];
      //node->parents[parent_idx]->children.push_back(node);
      //break;
    //}
  //}
//}

void ChangeParent(Node* node, int parent_idx, NodeNetwork &nn) {
  for (int i = 0; i < node->loc0; i++) {
  assert(nn.nodes[i].size() > 2); // If a layer has only 2 nodes, then there is nothing to change
  }
  while (true) {
    std::vector<int> row_idxs(node->loc0);
    std::iota(row_idxs.begin(), row_idxs.end(), 0);
    std::vector<int> choice_row;
    std::experimental::sample(row_idxs.begin(), row_idxs.end(), std::back_inserter(choice_row), 1, std::mt19937{std::random_device{}()});

    std::vector<int> col_idxs(nn.nodes[choice_row[0]].size());
    std::iota(col_idxs.begin(), col_idxs.end(), 0);
    std::vector<int> choice_col;
    std::experimental::sample(col_idxs.begin(), col_idxs.end(), std::back_inserter(choice_col), 1, std::mt19937{std::random_device{}()});

    if (!(choice_row[0] != node->parents[parent_idx]->loc0 &&
        choice_col[0] != node->parents[parent_idx]->loc1) &&
        !(choice_row[0] != node->parents[!parent_idx]->loc0 &&
        choice_col[0] != node->parents[!parent_idx]->loc1)) {
      node->parents[parent_idx]->children.erase(std::find(node->parents[parent_idx]->children.begin(),node->parents[parent_idx]->children.end(),node));
      node->parents[parent_idx] = &nn.nodes[choice_row[0]][choice_col[0]];
      node->parents[parent_idx]->children.push_back(node);
      break;
    }
  }
}

void SaveRun(NodeNetwork &nn, std::string out_str, std::string save_folder) {
  int file_cnt = 0;

  DIR *dir;
  struct dirent *diread;

  if ((dir = opendir(save_folder.c_str())) != nullptr) {
      while ((diread = readdir(dir)) != nullptr) {
          file_cnt += 1;
      }
      closedir(dir);
  } else {
      perror("Could not save AIG");
  }
  file_cnt -= 2; // First two are . and ..

  std::string num = std::to_string(file_cnt);
  std::string padding = "";
  while(padding.length() + num.length() < 4) {
      padding += "0";
  }
  std::string folder_path = save_folder + "/" + padding + num;
  mkdir(folder_path.c_str(), ACCESSPERMS);
  ExportAagRepr(nn, save_folder + "/" + padding + num + "/aig.aag");

  std::ofstream o;
  o.open(save_folder + "/" + padding + num + "/" + padding + num + ".log");
  o << out_str;
  o.close();
}

void SearchAroundNode(
  Node* node,
  NodeNetwork &nn,
  std::vector<bool> &pred,
  double &score,
  Node* old_parent,
  std::vector<double> &accuracies,
  std::vector<int> &mode,
  std::vector<Node*> &node_candidates,
  std::vector<Node*> &parent_candidates,
  std::vector<std::vector<bool>> &X_train,
  std::vector<bool> &y_train
  ) {
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

  /*
  // Change parent 0 full search
  old_parent = node->parents[0];
  for (int i = 0; i < node->loc0; i++) {
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      if (!(i != node->parents[0]->loc0 && j != node->parents[0]->loc1) &&
          !(i != node->parents[1]->loc0 && j != node->parents[1]->loc1)) {
        node->parents[0]->children.erase(std::find(node->parents[0]->children.begin(),node->parents[0]->children.end(),node));
        node->parents[0] = &nn.nodes[i][j];
        node->parents[0]->children.push_back(node);
      //
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
      }
    }
  }

  // Change parent 1 full search
  old_parent = node->parents[1];
  for (int i = 0; i < node->loc0; i++) {
    for (int j = 0; j < nn.nodes[i].size(); j++) {
      if (!(i != node->parents[0]->loc0 && j != node->parents[0]->loc1) &&
          !(i != node->parents[1]->loc0 && j != node->parents[1]->loc1)) {
        node->parents[1]->children.erase(std::find(node->parents[1]->children.begin(),node->parents[1]->children.end(),node));
        node->parents[1] = &nn.nodes[i][j];
        node->parents[1]->children.push_back(node);
      //
        pred = Predict(nn, X_train, "tmp");
        score = AccuracyScore(y_train, pred);
        accuracies.push_back(score);
        mode.push_back(4); // 3 stands for "change parent 0"
        node_candidates.push_back(node);
        parent_candidates.push_back(node->parents[1]);
        // Deregistering node from new parent
        node->parents[1]->children.erase(std::find(node->parents[1]->children.begin(),node->parents[1]->children.end(),node));
        // Setting parent to old parent
        node->parents[1] = old_parent;
        // Registering node at old parent again
        node->parents[1]->children.push_back(node);
      }
    }
  }
  */

  // 3: Change parent 0 heuristic search
#pragma omp critical
  {
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
  }

  // 4: Change parent 1 heuristic search
#pragma omp critical
  {
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
}
