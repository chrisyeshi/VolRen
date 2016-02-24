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

void VolRenRaycast::setFrustum(const std::shared_ptr<IRaycastFrustum> &frustum)
{
    int texWidth = _frustum->texWidth();
    int texHeight = _frustum->texHeight();
    int volWidth = _frustum->volWidth();
    int volHeight = _frustum->volHeight();
    int volDepth = _frustum->volDepth();
    QMatrix4x4 matView = _frustum->matView();
    QMatrix4x4 matProj = _frustum->matProj();

    _frustum = frustum;

    _frustum->setTexSize(texWidth, texHeight);
    _frustum->setVolSize(volWidth, volHeight, volDepth);
    _frustum->setMatVP(matView, matProj);
}

void VolRenRaycast::render()
{
    raycast();
}

} // namespace volren
} // namespace yy

