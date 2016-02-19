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
  , _frustum(new RaycastFrustum())
{

}

VolRenRaycast::~VolRenRaycast()
{
}

void VolRenRaycast::resize(int w, int h)
{
    _frustum->setTexSize(w, h);
}

void VolRenRaycast::setMatVP(const QMatrix4x4 &matView, const QMatrix4x4 &matProj)
{
    _frustum->setMatVP(matView, matProj);
}

void VolRenRaycast::render()
{
    raycast();
}

} // namespace volren
} // namespace yy

