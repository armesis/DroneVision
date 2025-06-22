#pragma once
#include <iostream>
#include <vector>
#include <cstdint>

namespace util
{
    void printShape(const std::vector<int64_t>& dims,
                    std::ostream& os = std::cout);
}
