#pragma once
#include <vector>
#include <string>

// COCO / custom labels exposed as a single read-only vector
const std::vector<std::string>& cocoClassNames();
