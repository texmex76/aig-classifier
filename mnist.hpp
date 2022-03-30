#pragma once

int ReverseInt(int i);

void ReadMNIST(int NumberOfImages, int DataOfAnImage,std::string MNISTPath, std::vector<std::vector<double>> &arr);

void ReadMNISTLabels(int NumberOfImages, std::string MNISTPath, std::vector<double> &arr);

void BinarizeMNISTLabels(std::vector<double> &labels, std::vector<bool> &binary_labels);

void BinarizeMNIST(std::vector<std::vector<double>> &arr, std::vector<std::vector<bool>> &binary_arr);
