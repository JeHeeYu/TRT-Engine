#include "model/model_utils.h"

int ModelUtils::getDimmensionChannel(nvinfer1::ITensor* tensor)
{
    nvinfer1::Dims dims = tensor->getDimensions();
    if (dims.nbDims == 3)      // Implicit Batch
        return dims.d[0];      // [C, H, W]
    else if (dims.nbDims == 4) // Explicit Batch
        return dims.d[1];      // [N, C, H, W]
    else
        throw std::runtime_error("Error tensor dimension : " + std::to_string(dims.nbDims));
}