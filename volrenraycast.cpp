#include "volrenraycast.h"
#include <map>
#include <cassert>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"
#include "volumegl.h"

namespace yy {
namespace volren {

VolRenRaycast::VolRenRaycast(const Method &method)
 : VolRen(method)
{

}

VolRenRaycast::~VolRenRaycast()
{
}

void VolRenRaycast::resize(int w, int h)
{
    _frustum.setTexSize(w, h);
}

void VolRenRaycast::render(const QMatrix4x4& v, const QMatrix4x4 &p)
{
    _frustum.setMatVP(v, p);
    // raycast
    raycast(_frustum.matModel(), v, p);
}

} // namespace volren
} // namespace yy

