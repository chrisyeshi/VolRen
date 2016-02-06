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

void VolRenRaycast::initializeGL()
{
    frustum.initializeGL();
}

void VolRenRaycast::resize(int w, int h)
{
    frustum.setResolution(w, h);
}

void VolRenRaycast::render(const QMatrix4x4& v, const QMatrix4x4 &p)
{
    frustum.entryTexture(v, p);
    frustum.exitTexture(v, p);
    // raycast
    raycast(frustum.modelMatrix(), v, p);
}

} // namespace volren
} // namespace yy

