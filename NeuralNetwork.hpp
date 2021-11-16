#ifndef BP2_NEURALNETWORKS_HPP
#define BP2_NEURALNETWORKS_HPP


#include "setting.h"


// 神经网络节点
class Node {
public:
    double value{};                 // 节点值
    double bias{};                  // 阈值
    double* weight = nullptr;       // 权值

public:
    void init(int num);             // 初始化
    ~Node();
};

/**
 * 构造下一层
 * @param num: 下一层节点个数
 * @return
 * */
void Node::init(int num) {
    weight = new double[num];

    mt19937 rd;
    rd.seed(random_device()());
    uniform_real_distribution<double> distribution(-0.1, 0.1);
    bias = distribution(rd);
    for (int i = 0; i < num; ++i) {
        weight[i] = distribution(rd);
    }
}

Node::~Node() {
    delete[] weight;
}


// 神经网络
class NeuralNetwork {
public:
    Node inputLayer[INPUT_NODE];                        // 输入层
    Node hiddenLayer[HIDDEN_NODE];                      // 隐藏层
    Node outputLayer[OUTPUT_NODE];                      // 输出层

    double y_hat[OUTPUT_NODE]{};                        // 计算值: y_hat
    double y[OUTPUT_NODE]{};                            // 实际值: y
public:
    NeuralNetwork();                                    // 构造函数
    double sigmoid(double x);                           // sigmoid函数
    double tanh(double x);                              // tanh函数
    double ReLU(double x);                              // ReLU函数
    double loss();                                      // 损失函数
    void forwardPropagating(const double* front, int f);// 正向传播
    void backPropagating(const double* behind, int f);  // 反向传播
    void printInfo(int times);                          // 输出信息

};

// 构造函数
NeuralNetwork::NeuralNetwork() {
    mt19937 rd;
    rd.seed(random_device()());
    uniform_real_distribution<double> distribution(-0.1, 0.1);


    // 初始化输入层到隐藏层权重
    for (auto& i : inputLayer) {
        i.init(HIDDEN_NODE);
    }

    // 初始化隐藏层到输出层权重
    for (auto& i : hiddenLayer) {
        i.init(OUTPUT_NODE);
    }

}

/**
 * 激活函数: sigmoid函数
 * @param x
 * @return 1.0 / (1 + exp(-x))
 * */
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1 + exp(-x));
}

/**
 * 激活函数: tanh函数
 * @param x
 * @return (exp(x) - exp(-x)) / (exp(x) + exp(-x))
 * */
double NeuralNetwork::tanh(double x) { 
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

/**
 * 激活函数: ReLU函数
 * @param x
 * @return max(0.0, x)
 * */
double NeuralNetwork::ReLU(double x) {
    return max(0.0, x);
}

/**
 * 损失函数
 * @return (y - y_hat)^2 / (2*OUTPUT_NODE)
 * */
double NeuralNetwork::loss() {
    double l = 0;
    for (int i = 0; i < OUTPUT_NODE; ++i) {
        l += (y[i] - y_hat[i]) * (y[i] - y_hat[i]);
    }
    return l / 2 / OUTPUT_NODE;
}


/**
 * 正向传播
 * @param front: 前层数据
 * */
void NeuralNetwork::forwardPropagating(const double* front, int f) {
    // 初始化输入层节点的值
    for (int i = 0; i < INPUT_NODE; ++i) {
        inputLayer[i].value = front[i];
    }

    // 通过激活函数计算隐藏层节点的值
    for (int i = 0; i < HIDDEN_NODE; ++i) {
        double x = 0;
        for (auto& j : inputLayer) {
            x += j.value * j.weight[i];
        }
        x -= hiddenLayer[i].bias;
        switch (f)
        {
        case 1:
            hiddenLayer[i].value = sigmoid(x);
            break;
        case 2:
            hiddenLayer[i].value = tanh(x);
            break;
        case 3:
            hiddenLayer[i].value = ReLU(x);
            break;
        default:
            break;
        }
    }

    // 通过激活函数计算输出层节点的值
    for (int i = 0; i < OUTPUT_NODE; ++i) {
        double x = 0;
        for (auto& j : hiddenLayer) {
            x += j.value * j.weight[i];
        }
        x -= outputLayer[i].bias;
        switch (f)
        {
        case 1:
            y_hat[i] = outputLayer[i].value = sigmoid(x);
            break;
        case 2:
            y_hat[i] = outputLayer[i].value = tanh(x);
            break;
        case 3:
            y_hat[i] = outputLayer[i].value = ReLU(x);
            break;
        default:
            break;
        }
        
    }
}

/**
 * 反向传播
 * @param behind: 后层数据
 * @param f: 选择激活函数  1: sigmoid    2: tanh    3: ReLU
 *
 * */
void NeuralNetwork::backPropagating(const double* behind, int f) {
    for (int i = 0; i < OUTPUT_NODE; ++i) {
        y[i] = behind[i];
    }

    // 更新输出层的阈值
    for (int i = 0; i < OUTPUT_NODE; ++i) {
        switch (f) {
        case 1:
            outputLayer[i].bias -= -RATE_SIGMOID * (y[i] - y_hat[i]) * y_hat[i] * (1 - y_hat[i]);
            break;
        case 2:
            outputLayer[i].bias -= -RATE_TANH * (y[i] - y_hat[i]) * (1 - y_hat[i] * y_hat[i]);
            break;
        case 3:
            outputLayer[i].bias -= -RATE_RELU * (y[i] - y_hat[i]);
            break;
        default:
            break;
        }
    }

    // 更新隐藏层权重
    for (auto& i : hiddenLayer) {
        for (int j = 0; j < OUTPUT_NODE; ++j) {
            switch (f) {
            case 1:
                i.weight[j] -= -RATE_SIGMOID * (y[j] - y_hat[j]) * y_hat[j] * (1 - y_hat[j]) * i.value;
                break;
            case 2:
                i.weight[j] -= -RATE_TANH * (y[j] - y_hat[j]) * (1 - y_hat[j] * y_hat[j]) * i.value;
                break;
            case 3:
                i.weight[j] -= -RATE_RELU * (y[j] - y_hat[j]) * i.value;
                break;
            default:
                break;
            }
        }
    }

    // 更新隐藏层阈值
    for (auto& i : hiddenLayer) {
        double delta = 0;
        switch (f) {
        case 1:
            for (int j = 0; j < OUTPUT_NODE; ++j) {
                delta += -(y[j] * y_hat[j]) * y_hat[j] * (1 - y_hat[j]) * i.weight[j];
            }
            i.bias -= RATE_SIGMOID * delta * i.value * (1 - i.value);
            break;
        case 2:
            for (int j = 0; j < OUTPUT_NODE; ++j) {
                delta += -(y[j] * y_hat[j]) * (1 - y_hat[j] * y_hat[j]) * i.weight[j];
            }
            i.bias -= RATE_TANH * delta * (1 - i.value * i.value);
            break;
        case 3:
            for (int j = 0; j < OUTPUT_NODE; ++j) {
                delta += -(y[j] * y_hat[j]) * i.weight[j];
            }
            i.bias -= RATE_RELU * delta;
            break;
        default:
            break;
        }

    }

    // 更新输入层权重
    for (auto& i : inputLayer) {
        for (int j = 0; j < HIDDEN_NODE; ++j) {
            double h = hiddenLayer[j].value;
            double x = i.value;
            double loss = 0;
            switch (f) {
            case 1:
                for (int k = 0; k < OUTPUT_NODE; ++k) {
                    loss += -(y[k] - y_hat[k]) * y_hat[k] * (1 - y_hat[k]) * hiddenLayer[j].weight[k];
                }
                i.weight[j] -= RATE_SIGMOID * loss * h * (1 - h) * x;
                break;
            case 2:
                for (int k = 0; k < OUTPUT_NODE; ++k) {
                    loss += -(y[k] - y_hat[k]) * (1 - y_hat[k] * y_hat[k]) * hiddenLayer[j].weight[k];
                }
                i.weight[j] -= RATE_TANH * loss * (1 - h * h) * x;
                break;
            case 3:
                for (int k = 0; k < OUTPUT_NODE; ++k) {
                    loss += -(y[k] - y_hat[k]) * hiddenLayer[j].weight[k];
                }
                i.weight[j] -= RATE_RELU * loss * x;
            default:
                break;
            }
        }
    }
}

/**
 * 输出训练信息
 * @param times: 输入层数据
 * */
//void NeuralNetwork::printInfo(int times) {
//    double l = loss();
//    cout << "训练了" << times << "次" << endl;
//    cout << "loss = " << l << endl;
//    for (int i = 0; i < OUTPUT_NODE; ++i) {
//        cout << "输出" << i + 1 << ": " << y_hat[i] << endl;
//    }
//}

#endif // BP2_NEURALNETWORKS_HPP
