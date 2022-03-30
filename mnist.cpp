// Function taken from
// https://compvisionlab.wordpress.com/2014/01/01/c-code-for-reading-mnist-data-set/
#include <iostream>
#include <fstream>
#include <vector>
#include "mnist.hpp"

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

void ReadMNIST(int NumberOfImages, int DataOfAnImage, std::string MNISTPath, std::vector<std::vector<double>> &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
    std::ifstream MNISTFile (MNISTPath,std::ios::binary);
    if (MNISTFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        MNISTFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        MNISTFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images); MNISTFile.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        MNISTFile.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    MNISTFile.read((char*)&temp,sizeof(temp));
                    arr[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
}

void ReadMNISTLabels(int NumberOfImages, std::string MNISTPath, std::vector<double> &arr)
{
    arr.resize(NumberOfImages);
    std::ifstream MNISTFile (MNISTPath,std::ios::binary);
    if (MNISTFile.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        MNISTFile.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        MNISTFile.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            MNISTFile.read((char*)&temp,sizeof(temp));
            arr[i] = (double)temp;
        }
    }
}

void BinarizeMNISTLabels(std::vector<double> &labels, std::vector<bool> &binary_labels) {
    binary_labels.resize(labels.size());
    for (int i = 0; i < labels.size(); i++) {
        if (labels[i] < 5) {
            binary_labels[i] = false;
        }
        else {
            binary_labels[i] = true;
        }
    }
}

void BinarizeMNIST(std::vector<std::vector<double>> &arr, std::vector<std::vector<bool>> &binary_arr) {
    binary_arr.resize(arr.size(), std::vector<bool>(arr[0].size()));
    for (int i = 0; i < arr.size(); i++) {
        for (int j = 0; j < arr[i].size(); j++) {
            if (arr[i][j] < 127) {
                binary_arr[i][j] = false;
            }
            else {
                binary_arr[i][j] = true;
            }
        }
    }
}
