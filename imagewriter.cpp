#include "imagewriter.h"
#include "imagewriterraf.h"
#include "imageraf.h"

namespace yy {
namespace volren {

std::unique_ptr<ImageWriter> ImageWriter::create(std::shared_ptr<ImageAbstract> image)
{
    if (ImageAbstract::TYPE_PBO_RAF == image->type())
    {
        std::shared_ptr<ImageRAF> derived = std::dynamic_pointer_cast<ImageRAF>(image);
        return std::unique_ptr<ImageWriterRAF>(new ImageWriterRAF(derived));
    }
    return nullptr;
}

} // namespace volren
} // namespace yy
