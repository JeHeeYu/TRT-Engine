

#include "plugin_factory.h"

//PluginFactory::PluginFactory() : m_ReorgLayer{nullptr}, m_RegionLayer{nullptr}
//{
//    for (int i = 0; i < m_MaxLeakyLayers; ++i) m_LeakyReLULayers[i] = nullptr;
//}
//
//nvinfer1::IPlugin* PluginFactory::createPlugin(const char* layerName, const void* serialData,
//                                               size_t serialLength)
//{
//    assert(isPlugin(layerName));
//    if (std::string(layerName).find("leaky") != std::string::npos)
//    {
//        assert(m_LeakyReLUCount >= 0 && m_LeakyReLUCount <= m_MaxLeakyLayers);
//        assert(m_LeakyReLULayers[m_LeakyReLUCount] == nullptr);
//		/*m_LeakyReLULayers[m_LeakyReLUCount]
//			= unique_ptr_INvPlugin(nvinfer1::plugin::createPReLUPlugin(serialData, serialLength));*/
//        ++m_LeakyReLUCount;
//        return m_LeakyReLULayers[m_LeakyReLUCount - 1].get();
//    }
//    else if (std::string(layerName).find("reorg") != std::string::npos)
//    {
//        assert(m_ReorgLayer == nullptr);
//        /*m_ReorgLayer = unique_ptr_INvPlugin(
//            nvinfer1::plugin::createYOLOReorgPlugin(serialData, serialLength));*/
//        return m_ReorgLayer.get();
//    }
//    else if (std::string(layerName).find("region") != std::string::npos)
//    {
//        assert(m_RegionLayer == nullptr);
//		/*m_RegionLayer = unique_ptr_INvPlugin(
//			 nvinfer1::plugin::createYOLORegionPlugin(serialData, serialLength));*/
//        return m_RegionLayer.get();
//    }
//    else if (std::string(layerName).find("yolo") != std::string::npos)
//    {
//        assert(m_YoloLayerCount >= 0 && m_YoloLayerCount < m_MaxYoloLayers);
//        assert(m_YoloLayers[m_YoloLayerCount] == nullptr);
//        m_YoloLayers[m_YoloLayerCount]
//            = unique_ptr_IPlugin(new YoloLayerV3(serialData, serialLength));
//        ++m_YoloLayerCount;
//        return m_YoloLayers[m_YoloLayerCount - 1].get();
//    }
//    else
//    {
//        std::cerr << "ERROR: Unrecognised layer : " << layerName << std::endl;
//        assert(0);
//        return nullptr;
//    }
//}
//
//bool PluginFactory::isPlugin(const char* name)
//{
//    return ((std::string(name).find("leaky") != std::string::npos)
//            || (std::string(name).find("reorg") != std::string::npos)
//            || (std::string(name).find("region") != std::string::npos)
//            || (std::string(name).find("yolo") != std::string::npos));
//}
//
//void PluginFactory::destroy()
//{
//    m_ReorgLayer.reset();
//    m_RegionLayer.reset();
//
//    for (int i = 0; i < m_MaxLeakyLayers; ++i)
//    {
//        m_LeakyReLULayers[i].reset();
//    }
//
//    for (int i = 0; i < m_MaxYoloLayers; ++i)
//    {
//        m_YoloLayers[i].reset();
//    }
//
//    m_LeakyReLUCount = 0;
//    m_YoloLayerCount = 0;
//}

/******* Yolo Layer V3 *******/
/*****************************/
namespace nvinfer1
{
	YoloLayer::YoloLayer()
	{}

	YoloLayer::YoloLayer(const void* data, size_t length)
	{
		const char *d = static_cast<const char*>(data), *a = d;
		re(d, m_NumBoxes);
		re(d, m_NumClasses);
		re(d, _n_grid_h);
		re(d, _n_grid_w);
		re(d, m_OutputSize);
		assert(d = a + length);
	}
	void YoloLayer::serialize(void* buffer)const noexcept
	{
		char *d = static_cast<char*>(buffer), *a = d;
		wr(d, m_NumBoxes);
		wr(d, m_NumClasses);
		wr(d, _n_grid_h);
		wr(d, _n_grid_w);
		wr(d, m_OutputSize);
		assert(d == a + getSerializationSize());
	}

	bool YoloLayer::supportsFormat(DataType type, PluginFormat format) const noexcept
	{
		return (type == DataType::kFLOAT && format == PluginFormat::kLINEAR);
	}

	void YoloLayer::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) noexcept
	{

	}

	IPluginV2* YoloLayer::clone() const noexcept
	{
		YoloLayer *p = new YoloLayer(m_NumBoxes,m_NumClasses,_n_grid_h,_n_grid_w);
		p->setPluginNamespace(_s_plugin_namespace.c_str());
		return p;
	}

	YoloLayer::YoloLayer(const uint32_t& numBoxes, const uint32_t& numClasses, const uint32_t& grid_h_, const uint32_t &grid_w_) :
		m_NumBoxes(numBoxes),
		m_NumClasses(numClasses),
		_n_grid_h(grid_h_),
		_n_grid_w(grid_w_)
	{
		assert(m_NumBoxes > 0);
		assert(m_NumClasses > 0);
		assert(_n_grid_h > 0);
		assert(_n_grid_w > 0);
		m_OutputSize = _n_grid_h * _n_grid_w * (m_NumBoxes * (4 + 1 + m_NumClasses));
	}

	int YoloLayer::getNbOutputs() const noexcept { return 1; }

	nvinfer1::Dims YoloLayer::getOutputDimensions(int index, const nvinfer1::Dims* inputs,
		int nbInputDims) noexcept
	{
		assert(index == 0);
		assert(nbInputDims == 1);
		
		return inputs[0];
	}

	//void YoloLayerV3::configure(const nvinfer1::Dims* inputDims, int nbInputs,
	//                            const nvinfer1::Dims* outputDims, int nbOutputs, int maxBatchSize) noexcept
	//{
	//    assert(nbInputs == 1);
	//    assert(inputDims != nullptr);
	//}

	int YoloLayer::initialize() noexcept { return 0; }

	void YoloLayer::terminate() noexcept {}

	size_t YoloLayer::getWorkspaceSize(int maxBatchSize) const noexcept
	{
		return 0;
	}

	int YoloLayer::enqueue(int batchSize,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream) noexcept
	{
		NV_CUDA_CHECK(cudaYoloLayerV3(inputs[0], outputs[0], batchSize, _n_grid_h, _n_grid_w, m_NumClasses,
			m_NumBoxes, m_OutputSize, stream));
		return 0;
	}

	size_t YoloLayer::getSerializationSize()const noexcept
	{
		return sizeof(m_NumBoxes) + sizeof(m_NumClasses) + sizeof(_n_grid_w) + sizeof(_n_grid_h) + sizeof(m_OutputSize);
	}




	PluginFieldCollection YoloLayerPluginCreator::mFC{};
	std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

	YoloLayerPluginCreator::YoloLayerPluginCreator()
	{
		mPluginAttributes.clear();

		mFC.nbFields = mPluginAttributes.size();
		mFC.fields = mPluginAttributes.data();
	}

	const char* YoloLayerPluginCreator::getPluginName() const noexcept
	{
		return "YOLO_TRT";
	}

	const char* YoloLayerPluginCreator::getPluginVersion() const noexcept
	{
		return "1.0";
	}

	const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames()noexcept
	{
		return &mFC;
	}

	IPluginV2* YoloLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)noexcept
	{
		YoloLayer* obj = new YoloLayer();
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}

	IPluginV2* YoloLayerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)noexcept
	{
		// This object will be deleted when the network is destroyed, which will
		// call MishPlugin::destroy()
		YoloLayer* obj = new YoloLayer(serialData, serialLength);
		obj->setPluginNamespace(mNamespace.c_str());
		return obj;
	}


	void YoloLayerPluginCreator::setPluginNamespace(const char* libNamespace)noexcept
	{
		mNamespace = libNamespace;
	}

	const char* YoloLayerPluginCreator::getPluginNamespace() const noexcept
	{
		return mNamespace.c_str();
	}


}
