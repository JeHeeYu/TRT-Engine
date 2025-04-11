#include "engine.h"

const std::vector<int> ModelEngine::COCO_CLASS_IDS{
    1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

ModelEngine::ModelEngine()
{
    // configParser.GetConfigFile(MODEL_CONFIG_PATH, configBlocks);
    enginePath =  "ENGINE.engine";
    engine = nullptr;
}

ModelEngine::~ModelEngine()
{
    if (network) network->destroy();
    if (engine) engine->destroy();
    if (builder) builder->destroy();
    if (config) config->destroy();
    if (modelStream) modelStream->destroy();
}

void ModelEngine::CreateEngine()
{
    configParser.GetConfigFile(MODEL_CONFIG_PATH, configBlocks);
    GetConfigBlocks();

    checkModelFiles();

    std::vector<float> weights = loadWeights(YOLO_NETWORK);

    builder = nvinfer1::createInferBuilder(logger);
    config = builder->createBuilderConfig();
    // network = builder->createNetworkV2(nvinfer1::NetworkDefinitionCreationFlags::kEXPLICIT_BATCH);
    network = builder->createNetworkV2(0U);
    std::vector<nvinfer1::Weights> trtWeights;
    int weightPtr = 0;

    if(!builder || !config || !network) {
        throw std::runtime_error("Engine craete faield : " + std::string(MODEL_CONFIG_PATH));
    }

    nvinfer1::ITensor* data = network->addInput(
    "yolov3", nvinfer1::DataType::kFLOAT,
    nvinfer1::Dims{ 3,static_cast<int>(yoloConfig.inputChannel), static_cast<int>(yoloConfig.inputHeight),
                        static_cast<int>(yoloConfig.inputWidth) });

    if(data == nullptr) {
        throw std::runtime_error("Input tensor creation failed");
    }

    nvinfer1::Dims divDims{
        3,
        {static_cast<int>(yoloConfig.inputChannel), 
        static_cast<int>(yoloConfig.inputHeight), 
        static_cast<int>(yoloConfig.inputWidth)}
    };

    nvinfer1::Weights divWeights {  nvinfer1::DataType::kFLOAT, nullptr, static_cast<int64_t>(yoloConfig.inputSize) };
    float *divWeight = new float[yoloConfig.inputSize];

    for(uint32_t i = 0; i < yoloConfig.inputSize; i++) {
        divWeight[i] = 255.0;
    }

    divWeights.values = divWeight;
    trtWeights.push_back(divWeights);
    nvinfer1::IConstantLayer *constDivide = network->addConstant(divDims, divWeights);

    if(constDivide == nullptr) {
        throw std::runtime_error("Constant layer creation failed");
    }    

    nvinfer1::IElementWiseLayer *elementDivide = network->addElementWise(*data, *constDivide->getOutput(0), nvinfer1::ElementWiseOperation::kDIV);

    if(elementDivide == nullptr) {
        throw std::runtime_error("Elementwise layer creation failed");
    }

    nvinfer1::ITensor *previous = elementDivide->getOutput(0);
    std::vector<nvinfer1::ITensor *> tensorOutputs;
    int n_layer_wts_index = 0;
    int n_output = 3 * (_n_classes + 5);
    int channels = yoloConfig.inputChannel;
    uint32_t outputTensorCount = 0;
    // int channels = previous->getDimensions().d[0];

    for(uint32_t i = 0; i < configBlocks.size(); i++) {
        assert(ModelUtils::getDimmensionChannel(previous) == channels);
        std::string layerIndex = "(" + std::to_string(i) + ")";

        if(configBlocks[i].at("type") == "convolutional") {
            std::string inputVolume = utils::StringUtils::DimensionsToString(previous->getDimensions());
            nvinfer1::ILayer *out;
            std::string layerType;
            std::string activation;

            if(configBlocks.at(i).find("activation") != configBlocks.at(i).end()) {
                activation = configBlocks.at(i).at("activation");
            }

            if((configBlocks.at(i).find("batch_normalize") != configBlocks.at(i).end()) && ("leaky" == activation)) {
                out = netAddConvBNLeaky(i, configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, network);
                layerType = "conv-bn-leaky";
            }
            else if((configBlocks.at(i).find("batch_normalize") != configBlocks.at(i).end()) && ("mish" == activation))  {
                out = net_conv_bn_mish(i, configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, network);
                layerType = "conv-bn-mish";
            }
            else {
                out = netAddConvLinear(i, configBlocks.at(i), weights, trtWeights, weightPtr, channels, previous, network);
                layerType = "conv-linear";
            }
            previous = out->getOutput(0);
            assert(previous != nullptr);
            channels = ModelUtils::getDimmensionChannel(previous);
            std::string outputVolume = utils::StringUtils::DimensionsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
        }
        else if(configBlocks.at(i).at("type") == "shortcut") { 
            assert(configBlocks.at(i).at("activation") == "linear");
            assert(configBlocks.at(i).find("from") != configBlocks.at(i).end());
            int from = stoi(configBlocks.at(i).at("from"));

            std::string inputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            // check if indexes are correct
            assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
            assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
            assert(i + from - 1 < i - 2);

            nvinfer1::IElementWiseLayer* ew
                = network->addElementWise(*tensorOutputs[i - 2], *tensorOutputs[i + from - 1],
                                            nvinfer1::ElementWiseOperation::kSUM);

            assert(ew != nullptr);
            std::string ewLayerName = "shortcut_" + std::to_string(i);
            ew->setName(ewLayerName.c_str());
            previous = ew->getOutput(0);
            assert(previous != nullptr);
            std::string outputVolume = utils::StringUtils::DimensionsToString(previous->getDimensions());
            tensorOutputs.push_back(ew->getOutput(0));
        }
        else if (configBlocks.at(i).at("type") == "yolo") {
            nvinfer1::Dims prevTensorDims = previous->getDimensions();
           // assert(prevTensorDims.d[1] == prevTensorDims.d[2]);
            TensorInfo& curYoloTensor = tensorInfos.at(outputTensorCount);
            curYoloTensor.gridSize = prevTensorDims.d[1];
            curYoloTensor.gridHeight = prevTensorDims.d[1];
            curYoloTensor.gridWidth = prevTensorDims.d[2];
            curYoloTensor.stride = yoloConfig.inputWidth / curYoloTensor.gridSize;
            curYoloTensor.strideHeight = yoloConfig.inputHeight / curYoloTensor.gridHeight;
            curYoloTensor.strideWidth = yoloConfig.inputWidth / curYoloTensor.gridWidth;
            tensorInfos.at(outputTensorCount).volume = curYoloTensor.gridHeight
                * curYoloTensor.gridWidth
                * (curYoloTensor.numBBoxes * (5 + curYoloTensor.numClasses));
            std::string layerName = "yolo_" + std::to_string(outputTensorCount);
            curYoloTensor.blobName = layerName;
            nvinfer1::IPluginV2* yoloPlugin
                = new nvinfer1::YoloLayer(tensorInfos.at(outputTensorCount).numBBoxes,
                                  tensorInfos.at(outputTensorCount).numClasses,
                                  tensorInfos.at(outputTensorCount).gridHeight,
                                  tensorInfos.at(outputTensorCount).gridWidth);
            assert(yoloPlugin != nullptr);
            nvinfer1::IPluginV2Layer* yolo = network->addPluginV2(&previous, 1, *yoloPlugin);
			
            assert(yolo != nullptr);
            yolo->setName(layerName.c_str());
            std::string inputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            previous = yolo->getOutput(0);
            assert(previous != nullptr);
            previous->setName(layerName.c_str());
            std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            network->markOutput(*previous);
            channels = ModelUtils::getDimmensionChannel(previous);
            tensorOutputs.push_back(yolo->getOutput(0));
            ++outputTensorCount;
        }
        else if (configBlocks.at(i).at("type") == "route") {
            size_t found = configBlocks.at(i).at("layers").find(",");
            if (found != std::string::npos)//concate multi layers 
            {
				std::vector<int> vec_index = split_layer_index(configBlocks.at(i).at("layers"), ",");
				for (auto &ind_layer:vec_index)
				{
					if (ind_layer < 0)
					{
						ind_layer = static_cast<int>(tensorOutputs.size()) + ind_layer;
					}
					assert(ind_layer < static_cast<int>(tensorOutputs.size()) && ind_layer >= 0);
				}
                nvinfer1::Dims refDims = tensorOutputs[vec_index[0]]->getDimensions();

                // modify
                int refC = refDims.d[0];
                int refH = refDims.d[1];
                int refW = refDims.d[2];

                nvinfer1::ITensor** concatInputs = reinterpret_cast<nvinfer1::ITensor**>(malloc(sizeof(nvinfer1::ITensor*) * vec_index.size()));
                for (size_t ind = 0; ind < vec_index.size(); ++ind)
                {
                    nvinfer1::ITensor* tensor = tensorOutputs[vec_index[ind]];
                    nvinfer1::Dims dims = tensor->getDimensions();

                    if (dims.d[1] != refH || dims.d[2] != refW)
                    {
                        auto resize = network->addResize(*tensor);
                        resize->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
                        resize->setOutputDimensions(nvinfer1::Dims3{dims.d[0], refH, refW});
                        tensor = resize->getOutput(0);
                    }

                    concatInputs[ind] = tensor;
                }

                nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs, static_cast<int>(vec_index.size()));

                assert(concat != nullptr);
                std::string concatLayerName = "route_" + std::to_string(i - 1);
                concat->setName(concatLayerName.c_str());
                // concatenate along the channel dimension
                concat->setAxis(0);
                previous = concat->getOutput(0);
                assert(previous != nullptr);
				nvinfer1::Dims debug = previous->getDimensions();
                std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
				int nums = 0;
				for (auto &indx:vec_index)
				{
					nums += ModelUtils::getDimmensionChannel(tensorOutputs[indx]);
				}
				channels = nums;
                tensorOutputs.push_back(concat->getOutput(0));
            }
            else //single layer
            {
                int idx = std::stoi(utils::StringUtils::trim(configBlocks.at(i).at("layers")));
                if (idx < 0)
                {
                    idx = static_cast<int>(tensorOutputs.size()) + idx;
                }
                assert(idx < static_cast<int>(tensorOutputs.size()) && idx >= 0);

				//route
				if (configBlocks.at(i).find("groups") == configBlocks.at(i).end())
				{
					previous = tensorOutputs[idx];
					assert(previous != nullptr);
					std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
					// set the output volume depth
					channels = ModelUtils::getDimmensionChannel(tensorOutputs[idx]);
					tensorOutputs.push_back(tensorOutputs[idx]);

				}
				//yolov4-tiny route split layer
				else
				{
					if (configBlocks.at(i).find("group_id") == configBlocks.at(i).end())
					{
						assert(0);
					}
					int chunk_idx = std::stoi(utils::StringUtils::trim(configBlocks.at(i).at("group_id")));
					nvinfer1::ILayer* out = layer_split(i, tensorOutputs[idx], network);
					std::string inputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
					previous = out->getOutput(chunk_idx);
					assert(previous != nullptr);
					channels = ModelUtils::getDimmensionChannel(previous);
					std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
					tensorOutputs.push_back(out->getOutput(chunk_idx));
				}
            }
        }
        else if (configBlocks.at(i).at("type") == "upsample") {
            std::string inputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddUpsample(i - 1, configBlocks[i], weights, trtWeights,
                                                   channels, previous, network);
            previous = out->getOutput(0);
            std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
        }
        else if (configBlocks.at(i).at("type") == "maxpool") {
            // Add same padding layers
            if (configBlocks.at(i).at("size") == "2" && configBlocks.at(i).at("stride") == "1") {
                //  m_TinyMaxpoolPaddingFormula->addSamePaddingLayer("maxpool_" + std::to_string(i));
            }
            std::string inputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            nvinfer1::ILayer* out = netAddMaxpool(i, configBlocks.at(i), previous, network);
            previous = out->getOutput(0);
            assert(previous != nullptr);
            std::string outputVol = utils::StringUtils::DimensionsToString(previous->getDimensions());
            tensorOutputs.push_back(out->getOutput(0));
        }
        else if (configBlocks.at(i).at("type") == "net") {

        }
        else{
            std::cout << "Unsupported layer type --> \"" << configBlocks.at(i).at("type") << "\""
                      << std::endl;
            assert(0);
        }
    }

    if (static_cast<int>(weights.size()) != weightPtr) {
        std::cout << "Number of unused weights left : " << static_cast<int>(weights.size()) - weightPtr << std::endl;
        assert(0);
    }

    if(!std::filesystem::exists(MODEL_CONFIG_PATH)) {
        std::cout << "Using previously generated plan file located at " << enginePath
            << std::endl;
        AllDestroy(trtWeights);
        return;
    }
    
    builder->setMaxBatchSize(BATCH_SIZE);

    config->setMaxWorkspaceSize(1 << 20);
    // if (dataType == nvinfer1::DataType::kINT8)
    // {
    //     assert((calibrator != nullptr) && "Invalid calibrator for INT8 precision");
    //     config->setFlag(nvinfer1::BuilderFlag::kINT8);
    //     config->setInt8Calibrator(calibrator);
    // }
    // else if (dataType == nvinfer1::DataType::kHALF)
    // {
    //     config->setFlag(nvinfer1::BuilderFlag::kFP16);
    // }

    int nbLayers = network->getNbLayers();
    int layersOnDLA = 0;

    if(std::filesystem::exists(MODEL_CONFIG_PATH)) {
        std::cout << "Building the TensorRT Engine..." << std::endl;
        engine = builder->buildEngineWithConfig(*network, *config);
        assert(engine != nullptr);

        // Serialize the engine
        writePlanFileToDisk();
    }

    // destroy
    AllDestroy(trtWeights);
}

void ModelEngine::checkModelFiles() const
{
    if(!std::filesystem::exists(MODEL_CONFIG_PATH)) {
        throw std::runtime_error("Model config file not found: " + std::string(MODEL_CONFIG_PATH));
    }
    if(!std::filesystem::exists(MODEL_WEIGHT_PATH)) {
        throw std::runtime_error("Model weight file not found: " + std::string(MODEL_WEIGHT_PATH));
    }
}

std::vector<float> ModelEngine::loadWeights(const std::string& type) const
{
    std::cout << "Load weights file" << std::endl;
    std::ifstream file(MODEL_WEIGHT_PATH, std::ios_base::binary);

    if(!file.is_open()) {
        throw std::runtime_error("Failed to open weight file : " + std::string(MODEL_WEIGHT_PATH));
    }

    // Major version info
    file.ignore(4);
    std::array<uint8_t, 1> networkTypeBuffer{};
    file.read(reinterpret_cast<char*>(networkTypeBuffer.data()), 1);

    // Check minor version info
    if(networkTypeBuffer[0] == 1) {
        // Ex version >= 1.0 minor 3byte + revision 4byte + seen 4byte
        file.ignore(11);
    } 
    else if(networkTypeBuffer[0] == 2) {
        // Ex version >= 2.0 minor 3byte + revision 4byte + seen 8byte
        file.ignore(15);
    } 
    else {
        throw std::runtime_error("Invalid network type");
    }

    std::vector<float> weights;
    std::array<uint8_t, 4> floatBuffer{};
    
    while(file.read(reinterpret_cast<char*>(floatBuffer.data()), 4)) {
        float weight;
        std::memcpy(&weight, floatBuffer.data(), sizeof(float)); 
        weights.push_back(weight);
    }

    std::cout << "Loading complete!" << std::endl;
    return weights;
}

void ModelEngine::GetConfigBlocks()
{
    std::cout << "Get config blocks" << std::endl;

    for(auto block : configBlocks) {
        if(block.at("type") == "net") {
            if(block.find("height") == block.end() || 
                block.find("width") == block.end() || 
                block.find("channels") == block.end() || 
                block.find("batch") == block.end()) {
                throw std::runtime_error("Missing required parameters in network config");
            }

            yoloConfig.inputChannel = std::stoi(block.at("channels"));
            yoloConfig.inputWidth = std::stoi(block.at("width"));
            yoloConfig.inputHeight = std::stoi(block.at("height"));
            
            yoloConfig.inputSize = yoloConfig.inputChannel * yoloConfig.inputWidth * yoloConfig.inputHeight;
        }
        else if((block.at("type") == "region") ||(block.at("type") == "yolo")) {
            if(block.find("num") == block.end() || 
                block.find("classes") == block.end() || 
                block.find("anchors") == block.end()) {
                throw std::runtime_error("Missing required parameters in " + block.at("type") + " layer");
            }

            TensorInfo tensorInfo;
            std::string anchors = block.at("anchors");
            while(!anchors.empty()) {
                size_t npos = anchors.find(',');
                if(npos != std::string::npos) {
                    float anchor = std::stof(utils::StringUtils::trim(anchors.substr(0, npos)));
                    tensorInfo.anchors.push_back(anchor);
                    anchors.erase(0, npos + 1);
                }
                else {
                    float anchor = std::stof(utils::StringUtils::trim(anchors));
                    tensorInfo.anchors.push_back(anchor);
                    break;
                }
            }

            if((networkType == "yolov3") ||(networkType == "yolov3-tiny") ||
               (networkType == "yolov4") ||(networkType == "yolov4-tiny")) {
                if(block.find("mask") == block.end()) {
                    throw std::runtime_error("Missing 'mask' param in " + block.at("type") + " layer");
                }

                std::string maskString = block.at("mask");
                while(!maskString.empty()) {
                    size_t npos = maskString.find_first_of(',');

                    if(npos != std::string::npos) {
                        uint32_t mask = std::stoul(utils::StringUtils::trim(maskString.substr(0, npos)));
                        tensorInfo.masks.push_back(mask);
                        maskString.erase(0, npos + 1);
                    }
                    else {
                        uint32_t mask = std::stoul(utils::StringUtils::trim(maskString));
                        tensorInfo.masks.push_back(mask);
                        break;
                    }
                }
            }

            tensorInfo.numBBoxes = tensorInfo.masks.size() > 0 ? 
                tensorInfo.masks.size() : std::stoul(utils::StringUtils::trim(block.at("num")));
            tensorInfo.numClasses = std::stoul(utils::StringUtils::trim(block.at("classes")));

            if(yoloConfig.classNames.empty()) {
                for(uint32_t i = 0; i < tensorInfo.numClasses; i++) {
                    yoloConfig.classNames.push_back(std::to_string(i));
                }
            }

            static int index = 0;
            
            tensorInfo.blobName = "yolo_" + std::to_string(index);
            tensorInfo.gridSize = (yoloConfig.inputHeight / 32) * std::pow(2, index);
            tensorInfo.gridWidth = (yoloConfig.inputWidth / 32) * std::pow(2, index);
            tensorInfo.gridHeight = (yoloConfig.inputHeight / 32) * std::pow(2, index);
            tensorInfo.stride = yoloConfig.inputHeight / tensorInfo.gridSize;
            tensorInfo.strideHeight = yoloConfig.inputHeight / tensorInfo.gridHeight;
            tensorInfo.strideWidth = yoloConfig.inputWidth / tensorInfo.gridWidth;
            tensorInfo.volume = tensorInfo.gridHeight * tensorInfo.gridWidth
                                * (tensorInfo.numBBoxes * (tensorInfo.numClasses + 5));

            tensorInfos.push_back(tensorInfo);
            index++;
        }
    }
}







nvinfer1::ILayer* ModelEngine::netAddConvBNLeaky(int layerIdx,
									std::map<std::string, std::string>& block,
                                    std::vector<float>& weights,
                                    std::vector<nvinfer1::Weights>& trtWeights,
									int& weightPtr,
                                    int& inputChannels,
									nvinfer1::ITensor* input,
                                    nvinfer1::INetworkDefinition* network)
{
    if(block.at("type") != "convolutional") {
        throw std::runtime_error("Layer type must be convolutional");
    }
    if(block.find("batch_normalize") == block.end()) {
        throw std::runtime_error("Batch normalization not found");
    }
    if(block.at("batch_normalize") != "1") {
        throw std::runtime_error("Batch normalization must be enabled");
    }
    if(block.at("activation") != "leaky") {
        throw std::runtime_error("Activation must be leaky");
    }
    if(block.find("filters") == block.end()) {
        throw std::runtime_error("Filters parameter not found");
    }
    if(block.find("pad") == block.end()) {
        throw std::runtime_error("Pad parameter not found");
    }
    if(block.find("size") == block.end()) {
        throw std::runtime_error("Size parameter not found");
    }
    if(block.find("stride") == block.end()) {
        throw std::runtime_error("Stride parameter not found");
    }

    bool batchNormalize, bias;
    if (block.find("batch_normalize") != block.end())
    {
        batchNormalize = (block.at("batch_normalize") == "1");
        bias = false;
    }
    else
    {
        batchNormalize = false;
        bias = true;
    }
    // all conv_bn_leaky layers assume bias is false
    if(batchNormalize != true || bias != false) {
        throw std::runtime_error("Batch normalization must be enabled and bias must be false");
    }

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;

    /***** CONVOLUTION LAYER *****/
    /*****************************/
    // batch norm weights are before the conv layer
    // load BN biases (bn_biases)
    std::vector<float> bnBiases;
    for (int i = 0; i < filters; ++i)
    {
        bnBiases.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN weights
    std::vector<float> bnWeights;
    for (int i = 0; i < filters; ++i)
    {
        bnWeights.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_mean
    std::vector<float> bnRunningMean;
    for (int i = 0; i < filters; ++i)
    {
        bnRunningMean.push_back(weights[weightPtr]);
        weightPtr++;
    }
    // load BN running_var
    std::vector<float> bnRunningVar;
    for (int i = 0; i < filters; ++i)
    {
        // 1e-05 for numerical stability
        bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5f));
        weightPtr++;
    }
    // load Conv layer weights (GKCRS)
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, 0};
    trtWeights.push_back(convBias);
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input,
		filters,
		nvinfer1::DimsHW{kernelSize, kernelSize},
		convWt,
		convBias);
    if(conv == nullptr) {
        throw std::runtime_error("Convolution layer creation failed");
    }
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

    /***** BATCHNORM LAYER *****/
    /***************************/
    size = filters;
    // create the weights
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, size};
    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        shiftWt[i]
            = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    }
    shift.values = shiftWt;
    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    }
    scale.values = scaleWt;
    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i)
    {
        powerWt[i] = 1.0;
    }
    power.values = powerWt;
    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);
    // Add the batch norm layers
    nvinfer1::IScaleLayer* bn = network->addScale(
        *conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    if(bn == nullptr) {
        throw std::runtime_error("Batch normalization layer creation failed");
    }
    std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
    bn->setName(bnLayerName.c_str());
    /***** ACTIVATION LAYER *****/
    /****************************/
	auto leaky = network->addActivation(*bn->getOutput(0),nvinfer1::ActivationType::kLEAKY_RELU);
	leaky->setAlpha(0.1f);
	if(leaky == nullptr) {
        throw std::runtime_error("Leaky ReLU layer creation failed");
    }
	std::string leakyLayerName = "leaky_" + std::to_string(layerIdx);
	leaky->setName(leakyLayerName.c_str());

    return leaky;
}

nvinfer1::ILayer* ModelEngine::net_conv_bn_mish(int layerIdx, 
	std::map<std::string, std::string>& block,
	std::vector<float>& weights,
	std::vector<nvinfer1::Weights>& trtWeights,
	int& weightPtr,
	int& inputChannels,
	nvinfer1::ITensor* input,
	nvinfer1::INetworkDefinition* network)
{
	assert(block.at("type") == "convolutional");
	assert(block.find("batch_normalize") != block.end());
	assert(block.at("batch_normalize") == "1");
	assert(block.at("activation") == "mish");
	assert(block.find("filters") != block.end());
	assert(block.find("pad") != block.end());
	assert(block.find("size") != block.end());
	assert(block.find("stride") != block.end());

	bool batchNormalize, bias;
	if (block.find("batch_normalize") != block.end())
	{
		batchNormalize = (block.at("batch_normalize") == "1");
		bias = false;
	}
	else
	{
		batchNormalize = false;
		bias = true;
	}
	// all conv_bn_leaky layers assume bias is false
	assert(batchNormalize == true && bias == false);

	int filters = std::stoi(block.at("filters"));
	int padding = std::stoi(block.at("pad"));
	int kernelSize = std::stoi(block.at("size"));
	int stride = std::stoi(block.at("stride"));
	int pad;
	if (padding)
		pad = (kernelSize - 1) / 2;
	else
		pad = 0;

	/***** CONVOLUTION LAYER *****/
	/*****************************/
	// batch norm weights are before the conv layer
	// load BN biases (bn_biases)
	std::vector<float> bnBiases;
	for (int i = 0; i < filters; ++i)
	{
		bnBiases.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN weights
	std::vector<float> bnWeights;
	for (int i = 0; i < filters; ++i)
	{
		bnWeights.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_mean
	std::vector<float> bnRunningMean;
	for (int i = 0; i < filters; ++i)
	{
		bnRunningMean.push_back(weights[weightPtr]);
		weightPtr++;
	}
	// load BN running_var
	std::vector<float> bnRunningVar;
	for (int i = 0; i < filters; ++i)
	{
		// 1e-05 for numerical stability
		bnRunningVar.push_back(sqrt(weights[weightPtr] + 1.0e-5f));
		weightPtr++;
	}
	// load Conv layer weights (GKCRS)
	int size = filters * inputChannels * kernelSize * kernelSize;
	nvinfer1::Weights convWt{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* val = new float[size];
	for (int i = 0; i < size; ++i)
	{
		val[i] = weights[weightPtr];
		weightPtr++;
	}
	convWt.values = val;
	trtWeights.push_back(convWt);
	nvinfer1::Weights convBias{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	trtWeights.push_back(convBias);
	nvinfer1::IConvolutionLayer* conv = network->addConvolution(
		*input, filters, nvinfer1::DimsHW{ kernelSize, kernelSize }, convWt, convBias);
	assert(conv != nullptr);
	std::string convLayerName = "conv_" + std::to_string(layerIdx);
	conv->setName(convLayerName.c_str());
	conv->setStride(nvinfer1::DimsHW{ stride, stride });
	conv->setPadding(nvinfer1::DimsHW{ pad, pad });

	/***** BATCHNORM LAYER *****/
	/***************************/
	size = filters;
	// create the weights
	nvinfer1::Weights shift{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights scale{ nvinfer1::DataType::kFLOAT, nullptr, size };
	nvinfer1::Weights power{ nvinfer1::DataType::kFLOAT, nullptr, size };
	float* shiftWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		shiftWt[i]
			= bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
	}
	shift.values = shiftWt;
	float* scaleWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
	}
	scale.values = scaleWt;
	float* powerWt = new float[size];
	for (int i = 0; i < size; ++i)
	{
		powerWt[i] = 1.0;
	}
	power.values = powerWt;
	trtWeights.push_back(shift);
	trtWeights.push_back(scale);
	trtWeights.push_back(power);
	// Add the batch norm layers
	nvinfer1::IScaleLayer* bn = network->addScale(
		*conv->getOutput(0), nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
	assert(bn != nullptr);
	std::string bnLayerName = "batch_norm_" + std::to_string(layerIdx);
	bn->setName(bnLayerName.c_str());
	/***** ACTIVATION LAYER *****/
	/****************************/
	auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
	const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
	nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(layerIdx)).c_str(), pluginData);
	nvinfer1::ITensor* inputTensors[] = { bn->getOutput(0) };
	auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
	return mish;
}


nvinfer1::ILayer* ModelEngine::netAddConvLinear(int layerIdx, std::map<std::string, std::string>& block,
                                   std::vector<float>& weights,
                                   std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr,
                                   int& inputChannels, nvinfer1::ITensor* input,
                                   nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "convolutional");
    assert(block.find("batch_normalize") == block.end());
    assert(block.at("activation") == "linear");
    assert(block.find("filters") != block.end());
    assert(block.find("pad") != block.end());
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int filters = std::stoi(block.at("filters"));
    int padding = std::stoi(block.at("pad"));
    int kernelSize = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));
    int pad;
    if (padding)
        pad = (kernelSize - 1) / 2;
    else
        pad = 0;
    // load the convolution layer bias
    nvinfer1::Weights convBias{nvinfer1::DataType::kFLOAT, nullptr, filters};
    float* val = new float[filters];
    for (int i = 0; i < filters; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convBias.values = val;
    trtWeights.push_back(convBias);
    // load the convolutional layer weights
    int size = filters * inputChannels * kernelSize * kernelSize;
    nvinfer1::Weights convWt{nvinfer1::DataType::kFLOAT, nullptr, size};
    val = new float[size];
    for (int i = 0; i < size; ++i)
    {
        val[i] = weights[weightPtr];
        weightPtr++;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    nvinfer1::IConvolutionLayer* conv = network->addConvolution(
        *input, filters, nvinfer1::DimsHW{kernelSize, kernelSize}, convWt, convBias);
    assert(conv != nullptr);
    std::string convLayerName = "conv_" + std::to_string(layerIdx);
    conv->setName(convLayerName.c_str());
    conv->setStride(nvinfer1::DimsHW{stride, stride});
    conv->setPadding(nvinfer1::DimsHW{pad, pad});

    return conv;
}

std::vector<int> ModelEngine::split_layer_index(const std::string &s_,const std::string &delimiter_)
{
    std::vector<int> index;
    std::string s = s_;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter_)) != std::string::npos)
    {
        token = s.substr(0, pos);
        index.push_back(std::stoi(utils::StringUtils::trim(token)));
        s.erase(0, pos + delimiter_.length());
    }
    index.push_back(std::stoi(utils::StringUtils::trim(s)));
    return index;
}

nvinfer1::ILayer * ModelEngine::layer_split(const int n_layer_index_,
	nvinfer1::ITensor *input_,
	nvinfer1::INetworkDefinition* network)
{
	auto creator = getPluginRegistry()->getPluginCreator("CHUNK_TRT", "1.0");
	const nvinfer1::PluginFieldCollection* pluginData = creator->getFieldNames();
	nvinfer1::IPluginV2 *pluginObj = creator->createPlugin(("chunk" + std::to_string(n_layer_index_)).c_str(), pluginData);
	auto chunk = network->addPluginV2(&input_, 1, *pluginObj);
	return chunk;
}

nvinfer1::ILayer* ModelEngine::netAddUpsample(int layerIdx, std::map<std::string, std::string>& block,
                                 std::vector<float>& weights,
                                 std::vector<nvinfer1::Weights>& trtWeights, int& inputChannels,
                                 nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "upsample");
    nvinfer1::Dims inpDims = input->getDimensions();
    assert(inpDims.nbDims == 3);
   // assert(inpDims.d[1] == inpDims.d[2]);
    int n_scale = std::stoi(block.at("stride"));

	int c1 = inpDims.d[0];
	float *deval = new float[c1*n_scale*n_scale];
	for (int i = 0; i < c1*n_scale*n_scale; i++)
	{
		deval[i] = 1.0;
	}
	nvinfer1::Weights wts{ nvinfer1::DataType::kFLOAT, deval, c1*n_scale*n_scale };
	nvinfer1::Weights bias{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
	nvinfer1::IDeconvolutionLayer* upsample = network->addDeconvolutionNd(*input, c1, nvinfer1::DimsHW{ n_scale, n_scale }, wts, bias);
	upsample->setStrideNd(nvinfer1::DimsHW{ n_scale, n_scale });
	upsample->setNbGroups(c1);
	return upsample;
}


nvinfer1::ILayer* ModelEngine::netAddMaxpool(int layerIdx, std::map<std::string, std::string>& block,
                                nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network)
{
    assert(block.at("type") == "maxpool");
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* pool
        = network->addPoolingNd(*input, nvinfer1::PoolingType::kMAX, nvinfer1::DimsHW{size, size});
    assert(pool);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
	int pad = (size - 1) / 2;
	pool->setPaddingNd(nvinfer1::DimsHW{pad,pad});
    pool->setStrideNd(nvinfer1::DimsHW{stride, stride});
    pool->setName(maxpoolLayerName.c_str());

    return pool;
}

void ModelEngine::AllDestroy(std::vector<nvinfer1::Weights>& trtWeights)
{
    if (network) network->destroy();
    if (engine) engine->destroy();
    if (builder) builder->destroy();
    if (config) config->destroy();
    if (modelStream) modelStream->destroy();

    for(auto & trtWeight : trtWeights) {
        if (trtWeight.count > 0) free(const_cast<void*>(trtWeight.values));
    }
}


void ModelEngine::writePlanFileToDisk()
{
    std::cout << "Serializing the TensorRT Engine..." << std::endl;
    assert(engine && "Invalid TensorRT Engine");
    modelStream = engine->serialize();
    assert(modelStream && "Unable to serialize engine");
    assert(!enginePath.empty() && "Enginepath is empty");

    // write data to output file
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write(static_cast<const char*>(modelStream->data()), modelStream->size());
    std::ofstream outFile;
    outFile.open(enginePath, std::ios::binary | std::ios::out);
    outFile << gieModelStream.rdbuf();
    outFile.close();

    std::cout << "Serialized plan file cached at location : " << enginePath << std::endl;
}