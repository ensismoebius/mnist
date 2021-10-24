#include "mnist.h"
#include <fstream>
#include <opencv2/opencv.hpp>
namespace mnist
{
    inline int reverseInt(int i)
    {
        unsigned char ch1, ch2, ch3, ch4;
        ch1 = i & 255;
        ch2 = (i >> 8) & 255;
        ch3 = (i >> 16) & 255;
        ch4 = (i >> 24) & 255;
        return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
    }

    void readMnistImages(std::string filename, std::vector<cv::Mat> &vec)
    {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            int number_of_images = 0;
            int n_rows = 0;
            int n_cols = 0;

            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            file.read((char *)&number_of_images, sizeof(number_of_images));
            number_of_images = reverseInt(number_of_images);

            file.read((char *)&n_rows, sizeof(n_rows));
            n_rows = reverseInt(n_rows);

            file.read((char *)&n_cols, sizeof(n_cols));
            n_cols = reverseInt(n_cols);

            for (int i = 0; i < number_of_images; ++i)
            {
                cv::Mat tp = cv::Mat::zeros(n_rows, n_cols, CV_8UC1);
                for (int r = 0; r < n_rows; ++r)
                {
                    for (int c = 0; c < n_cols; ++c)
                    {
                        unsigned char temp = 0;
                        file.read((char *)&temp, sizeof(temp));
                        tp.at<uchar>(r, c) = (int)temp;
                    }
                }
                vec.push_back(tp);
            }
        }
    }

    void readMnistLabels(std::string full_path, std::vector<char> &labels)
    {
        int numberOfLabels;

        std::ifstream file(full_path, std::ios::binary);

        if (file.is_open())
        {
            int magic_number = 0;
            file.read((char *)&magic_number, sizeof(magic_number));
            magic_number = reverseInt(magic_number);

            if (magic_number != 2049)
                throw std::runtime_error("Invalid MNIST label file!");

            file.read((char *)&numberOfLabels, sizeof(numberOfLabels));
            numberOfLabels = reverseInt(numberOfLabels);

            labels.resize(numberOfLabels);

            for (int i = 0; i < numberOfLabels; i++)
            {
                file.read((char *)&labels[i], 1);
            }
        }
        else
        {
            throw std::runtime_error("Unable to open file `" + full_path + "`!");
        }
    }
}