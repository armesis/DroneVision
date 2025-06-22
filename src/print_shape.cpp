#include "print_shape.hpp"

namespace util
{
    void printShape(const std::vector<int64_t>& dims, std::ostream& os)
    {
        os << '[';
        for (size_t i = 0; i < dims.size(); ++i)
        {
            os << dims[i];
            if (i + 1 != dims.size()) os << 'x';
        }
        os << ']';
    }
}
