#ifndef BP2_GETIMAGE_HPP
#define BP2_GETIMAGE_HPP

#include "setting.h"
#include <fstream>

using namespace std;

class Image//单张图像
{
public:
    unsigned char tag;
    double data[INPUT_NODE];
    double label[OUTPUT_NODE];
};


class getImage {
public:
    Image* imageData;

public:
    void getTrainData(const char* dataPath, const char* labelPath);

    ~getImage();
};

void getImage::getTrainData(const char* dataPath, const char* labelPath) {
    // 读取图片
    unsigned char readBuffer[4];
    ifstream inFileData(dataPath, ios::in | ios::binary);
    if (!inFileData) {
        cout << "文件打开失败" << endl;
        return;
    }

    inFileData.read((char*)&readBuffer, 4);
    inFileData.read((char*)&readBuffer, 4);
    int count = (readBuffer[0] << 24) + (readBuffer[1] << 16) + (readBuffer[2] << 8) + readBuffer[3];//图像个数
    inFileData.read((char*)&readBuffer, 4);
    inFileData.read((char*)&readBuffer, 4);
    imageData = new Image[count];
    auto* data = new unsigned char[INPUT_NODE];
    for (int i = 0; i < count; ++i) {
        inFileData.read(reinterpret_cast<char*>(&data[0]), INPUT_NODE);
        for (int px = 0; px < INPUT_NODE; ++px) {
            imageData[i].data[px] = data[px] / (double)255 * 0.99 + 0.01;
        }

    }

    delete[] data;


    ifstream inFileLabel(labelPath, ios::in | ios::binary);
    inFileLabel.read((char*)&readBuffer, 4);
    inFileLabel.read((char*)&readBuffer, 4);
    count = (readBuffer[0] << 24) + (readBuffer[1] << 16) + (readBuffer[2] << 8) + readBuffer[3];//图像个数
    for (int i = 0; i < count; ++i) {
        inFileLabel.read((char*)readBuffer, 1);
        imageData[i].tag = readBuffer[0];

        for (double& j : imageData[i].label) {
            j = 0.01;
        }
        imageData[i].label[imageData[i].tag] = 0.99;
    }
    inFileLabel.close();
}

getImage::~getImage() {
    delete[]imageData;
}

#endif //BP2_GETIMAGE_HPP
