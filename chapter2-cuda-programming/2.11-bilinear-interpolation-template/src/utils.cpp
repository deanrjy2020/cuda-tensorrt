#include "utils.hpp"

#include <filesystem>

// Return only the file name without its extension, including for absolute paths.
std::string getFileName(std::string file_path) {
    return std::filesystem::path(file_path).stem().string();
}
