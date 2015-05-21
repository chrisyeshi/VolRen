#include "volren.h"
#include <cassert>
#include "volrenraycastgl.h"
#include "volrenraycastcuda.h"
#include "volrenraycastraf.h"

namespace yy {
namespace volren {

std::unique_ptr<VolRen> VolRen::create(const Method &method)
{
    assert(Method_Unknown != method);
    if (Method_Raycast_GL == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastGL());
    if (Method_Raycast_CUDA == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastCuda());
    if (Method_Raycast_RAF == method)
        return std::unique_ptr<VolRen>(new VolRenRaycastRAF());
    return nullptr;
}

} // namespace volren
} // namespace yy
