#ifndef MODEL_UTILS_H
#define MODEL_UTILS_H

#include <stdexcept>

#include "NvInfer.h"

class ModelUtils {
public:
    static int getDimmensionChannel(nvinfer1::ITensor* tensor);
};

#endif // MODEL_UTILS_H