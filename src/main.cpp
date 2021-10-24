#include <iostream>
#include <SFML/Graphics.hpp>

#include "lib/mnist.h"
#include "lib/NeuralNetwork.h"

void to_arma(const cv::Mat &src, NeuralNetwork::TrainningSample *dst)
{
    for (size_t i = 0; i < src.rows; i++)
    {
        for (size_t j = 0; j < src.cols; j++)
        {
            dst->input[i * src.cols + j] = std::isnan(src.at<float>(i, j)) ? 0 : src.at<float>(i, j);
        }
    }
}

// The activation function
float activationF(float value)
{
    return 1.0 / (1 + std::exp(-value));
}

// The derivative of activation function
float activationFD(float value)
{
    return value * (1 - value);
}

int main()
{
    // read MNIST iamge into OpenCV Mat vector
    std::vector<char> labels;
    std::vector<cv::Mat> images;
    std::vector<NeuralNetwork::TrainningSample *> examples;

    unsigned int inputs;
    unsigned int outputs;

    mnist::readMnistLabels("/home/ensismoebius/workspaces/c-workspace/mnist/data/train-labels.idx1-ubyte", labels);
    mnist::readMnistImages("/home/ensismoebius/workspaces/c-workspace/mnist/data/train-images.idx3-ubyte", images);

    std::cout << images.size() << std::endl;
    std::cout << labels.size() << std::endl;

    inputs = images[0].rows * images[0].cols;
    outputs = 1;

    examples.resize(images.size());

    for (int i = 0; i < images.size(); i++)
    {
        examples[i] = new NeuralNetwork::TrainningSample(inputs, outputs);
        examples[i]->target[0] = float(labels[i]);

        to_arma(images[i], examples[i]);
    }

    NeuralNetwork::NeuralNetwork nn(inputs, 10, outputs);
    nn.setActivationFunction(activationF);
    nn.setActivationFunctionDerivative(activationFD);

    for (int i = 0; i < images.size(); i++)
    {
        std::cout << std::to_string(labels[i]) << std::endl;
        // cv::imshow("Window - float", images[i]);

        nn.train(*examples[i], 0.1);

        // cv::waitKey(1);
    }

    std::cout << nn.classify(examples[0]->input).at(0, 0) << std::endl;

    return 0;
}