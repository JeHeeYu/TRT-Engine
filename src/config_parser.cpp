#include "config_parser.h"

ConfigParser::ConfigParser()
{

}

ConfigParser::~ConfigParser()
{

}

void ConfigParser::GetConfigFile(const std::string& configPath, std::vector<std::map<std::string, std::string>>& blocks) {
    if(!std::filesystem::exists(configPath)) {
        throw std::runtime_error("Model config file not found : " + std::string(MODEL_CONFIG_PATH));
    }

    std::ifstream file(configPath);

    if(!file.is_open()) {
        throw std::runtime_error("Failed to open config file : " + configPath);
    }

    std::string line;
    std::map<std::string, std::string> block;

    while(std::getline(file, line)) {
        line = utils::StringUtils::trim(line);

        if(line.empty() || line.front() == '#') {
            continue;
        }
        
        if(line.front() == '[') {
            if (!block.empty()) {
                blocks.push_back(block);
                block.clear();
            }

            std::string key = "type";
            std::string value = utils::StringUtils::trim(line.substr(1, line.size() - 2));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
        else {
            size_t cpos = line.find('=');
            std::string key = utils::StringUtils::trim(line.substr(0, cpos));
            std::string value = utils::StringUtils::trim(line.substr(cpos + 1));
            block.insert(std::pair<std::string, std::string>(key, value));
        }
    }

    blocks.push_back(block);
}
