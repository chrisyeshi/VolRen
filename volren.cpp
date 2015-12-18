#include "volren.h"
#include <cassert>
#include "volrenraycastgl.h"
#ifdef ENABLE_CUDA
    #include "volrenraycastcuda.h"
    #include "volrenraycastraf.h"
#endif // ENABLE_CUDA

namespace yy {
namespace volren {

// TODO: put into a VolRenFactory and use a map for the creators,
// C++ doesn't support static initialization in a static library (lame)
std::unique_ptr<VolRen> VolRen::create(const Method &method)
{
    assert(Method_Unknown != method);
    if (Method_Raycast_GL == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastGL());
#ifdef ENABLE_CUDA
    if (Method_Raycast_CUDA == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastCuda());
    if (Method_Raycast_RAF == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastRAF());
#endif // ENABLE_CUDA
    return nullptr;
}

} // namespace volren
} // namespace yy
