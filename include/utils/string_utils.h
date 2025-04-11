#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <string>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <sstream>

#include "NvInfer.h"

namespace utils {

class StringUtils {
public:
    static std::string trim(const std::string& str);
    static std::string DimensionsToString(const nvinfer1::Dims Dimensions);
};

}

#endif // STRING_UTILS_H