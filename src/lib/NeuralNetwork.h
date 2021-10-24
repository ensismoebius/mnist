/**
 * @author André Furlan
 * @email ensismoebius@gmail.com
 * This whole project are under GPLv3, for
 * more information read the license file
 *
 * @date 2021-10-18
 */

#ifndef src_lib_NeuralNetwork_h
#define src_lib_NeuralNetwork_h

#include <armadillo>

namespace NeuralNetwork
{
#define m arma::Mat<float>

    class TrainningSample
    {
    public:
        std::vector<float> target;
        std::vector<float> input;
        TrainningSample(unsigned int inputs, unsigned int targets) : //
                                                                     input(inputs),
                                                                     target(targets)
        {
        }
    };

    class NeuralNetwork
    {
    private:
        unsigned int inputNodes;
        unsigned int hiddenNodes;
        unsigned int outputNodes;

        m hidden;
        m hiddenErrors;
        m weightsInputToHidden;

        m output;
        m outputErrors;
        m weightsHiddenToOutput;

        m biasHidden;
        m biasOutput;

        void initializesWeights();

        float (*activationFunction)(float value);
        float (*activationFunctionDerivative)(float value);

        m activationFunc(m value);
        m activationFuncD(m value);

    public:
        NeuralNetwork(unsigned int inputNodes, unsigned int hiddenNodes, unsigned int outputNodes);
        m classify(std::vector<float> input);
        void train(TrainningSample sample, float learnningRate);
        void setActivationFunction(float (*activationFunction)(float value));
        void setActivationFunctionDerivative(float (*activationFunctionDerivative)(float value));
    };

} // namespace NeuralNetwork
#endif