#ifndef IMAGEWRITER
#define IMAGEWRITER

#include <memory>
#include <unordered_map>
#include "imageabstract.h"

namespace yy {
namespace volren {

class ImageWriter
{
public:
    static std::unique_ptr<ImageWriter> create(std::shared_ptr<ImageAbstract> image);
    ImageWriter() {}
    virtual ~ImageWriter() {}

    virtual void write(const std::string& filename) const = 0;
};

} // namespace volren
} // namespace yy

#endif // IMAGEWRITER

