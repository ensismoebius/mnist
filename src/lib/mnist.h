/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * @date 2021-10-23
 */

#ifndef src_lib_mnist_h
#define src_lib_mnist_h

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace mnist
{
    void readMnistImages(std::string filename, std::vector<cv::Mat> &vec);

    void readMnistLabels(std::string full_path, std::vector<char> &labels);
}

#endif