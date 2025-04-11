#ifndef ENGINE_H
#define ENGINE_H

#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <cassert>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "NvInferPlugin.h"
#include "NvInferRuntimeCommon.h"

#include "config_parser.h"
#include "common.h"
#include "plugin_factory.h"
#include "model/model_utils.h"
#define BATCH_SIZE 1

typedef struct _YoloConfig
{
    uint32_t inputChannel;
    uint32_t inputWidth;
    uint32_t inputHeight;
    uint32_t inputSize;
    int batchSize;
    std::vector<std::string> classNames;
} YoloConfig;

typedef struct _TensorInfo
{
    std::string blobName;
    uint32_t stride{ 0 };
    uint32_t strideWidth{ 0 };
    uint32_t strideHeight{ 0 };
    uint32_t gridSize{ 0 };
	uint32_t gridWidth{ 0 };
	uint32_t gridHeight{ 0 };
    uint32_t numClasses{ 0 };
    uint32_t numBBoxes{ 0 };
    uint64_t volume{ 0 };
    std::vector<uint32_t> masks;
    std::vector<float> anchors;
    int bindingIndex{ -1 };
    float* hostBuffer{ nullptr };
} TensorInfo;

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING) : mSeverity(severity) {}
    
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= mSeverity) {
            switch (severity) {
                case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
                case Severity::kERROR: std::cerr << "ERROR: "; break;
                case Severity::kWARNING: std::cerr << "WARNING: "; break;
                case Severity::kINFO: std::cerr << "INFO: "; break;
                default: std::cerr << "UNKNOWN: "; break;
            }
            std::cerr << msg << std::endl;
        }
    }

private:
    Severity mSeverity;
};

class ModelEngine
{
public:
    ModelEngine();
    ~ModelEngine();

    void CreateEngine();
    std::vector<float> loadWeights(const std::string& type) const;

private:
    void checkModelFiles() const;
    void GetConfigBlocks();
    void AllDestroy(std::vector<nvinfer1::Weights>& trtWeights);


    nvinfer1::ILayer* netAddConvBNLeaky(int layerIdx,
									std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights,
									int& weightPtr,
                                    int& inputChannels,
									nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network);

    nvinfer1::ILayer* net_conv_bn_mish(int layerIdx, 
	std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights,
	int& weightPtr,
	int& inputChannels,
	nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network);



    nvinfer1::ILayer* netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network);

    std::vector<int> split_layer_index(const std::string &s_,const std::string &delimiter_);
    nvinfer1::ILayer * layer_split(const int n_layer_index_,
	nvinfer1::ITensor *input_,
	nvinfer1::INetworkDefinition* network);
    nvinfer1::ILayer* netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
    nvinfer1::ILayer* netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network);
void destroyNetworkUtils(std::vector<nvinfer1::Weights>& trtWeights);
    void writePlanFileToDisk();


    // COCO dataset class IDs
    static const std::vector<int> COCO_CLASS_IDS;

private:
    nvinfer1::IBuilder *builder;
    nvinfer1::INetworkDefinition *network;
    nvinfer1::IBuilderConfig *config;
    std::string networkType;
    Logger logger;
    ConfigParser configParser;
    YoloConfig yoloConfig;
    std::vector<std::map<std::string, std::string>> configBlocks;
    std::vector<TensorInfo> tensorInfos;
    std::string enginePath;
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IHostMemory* modelStream;

    // changed
    uint32_t _n_classes = 0;
};

#endif // ENGINE_H