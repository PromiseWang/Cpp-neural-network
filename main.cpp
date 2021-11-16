#include "Setting.h"
#include "NeuralNetwork.hpp"
#include "GetImage.hpp"

void AccuracyRate(int time, NeuralNetwork* network, getImage* image, int f)     //精确率评估
{
    int right = 0;                                                              //正确个数统计
    for (int count = 0; count < 10000; count++) {
        network->forwardPropagating(image->imageData[count].data, f);           //前向传播
        double value = -100;
        int tag = -100;
        for (int i = 0; i < 10; i++) {
            if (network->outputLayer[i].value > value) {
                value = network->outputLayer[i].value;
                tag = i;
            }
        }
        if (image->imageData[count].tag == tag) {
            right++;
        }
    }

    cout << "第" << time + 1 << "轮:  ";
    cout << "正确率为:" << right * 1.0 / 10000 << endl;
}

int main() {
    getImage trainData{}, testData{};
    trainData.getTrainData("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
    testData.getTrainData("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
    NeuralNetwork network;                                                      //定义神经网络
    vector<string> function = { "sigmoid", "tanh", "ReLU" };
    for (int f = 3; f <= 3; f++) {
        cout << "--------------------使用" << function[f - 1] << "作为激活函数开始训练--------------------" << endl;
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < 60000; i++) {
                network.forwardPropagating(trainData.imageData[i].data, f);     //前向传播
                network.backPropagating(trainData.imageData[i].label, f);       //反向传播
            }
            AccuracyRate(j, &network, &testData, f);

            // network.printInfo(j);
        }
        std::cout << "--------------------" << function[f - 1] << "训练完成!!--------------------" << endl;
    }

    system("PAUSE");
    return 0;
}