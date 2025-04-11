#include "utils/string_utils.h"
#include <algorithm>
#include <cctype>

namespace utils {

std::string StringUtils::trim(const std::string& str) {
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
        return std::isspace(c);
    });
    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();
    return (start < end) ? std::string(start, end) : std::string();
}

std::string StringUtils::DimensionsToString(const nvinfer1::Dims Dimensions)
{
    std::stringstream stream;
    if(!Dimensions.nbDims >= 1) {
        throw std::runtime_error("dimsToString failed : dims.nbDims = " + std::to_string(Dimensions.nbDims));
    }
    
    for (int i = 0; i < Dimensions.nbDims - 1; i++)
    {
        stream << std::setw(4) << Dimensions.d[i] << " x";
    }
    stream << std::setw(4) << Dimensions.d[Dimensions.nbDims - 1];

    return stream.str();
}

}