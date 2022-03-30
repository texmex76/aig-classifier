#include "ml.hpp"

double AccuracyScore(std::vector<bool> &y_true, std::vector<bool> &y_pred) {
  double corr = 0.0;
  for (int i = 0; i < y_true.size(); i++) {
    if (y_true[i] == y_pred[i]) {
      corr += 1;
    }
  }
  return corr / y_true.size();
}
