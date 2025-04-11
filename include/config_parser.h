#ifndef CONFIG_PARSER_H
#define CONFIG_PARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <map>
#include <algorithm>
#include <cctype>

#include "common.h"
#include "utils/string_utils.h"

class ConfigParser
{
public:
    ConfigParser();
    ~ConfigParser();

    void GetConfigFile(const std::string& configPath, std::vector<std::map<std::string, std::string>>& blocks);    
    
private:
    std::string configPath;

};


#endif // CONFIG_PARSER_H