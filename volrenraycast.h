#ifndef VOLRENRAYCAST_H
#define VOLRENRAYCAST_H

#include "volren.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "raycastfrustum.h"

class ImageAbstract;

namespace yy {
namespace volren {

class VolRenRaycast : public VolRen
{
public:
    VolRenRaycast(const Method& method);
    virtual ~VolRenRaycast();

    virtual void initializeGL() {}
    virtual void resize(int w, int h);
    virtual void setMatVP(const QMatrix4x4& matView, const QMatrix4x4& matProj);
    virtual void setFrustum(const std::shared_ptr<IRaycastFrustum> &frustum) { _frustum = frustum; }
    virtual void render();
    virtual std::shared_ptr<ImageAbstract> output() const = 0;

protected:
    virtual void raycast() = 0;

protected:
    std::shared_ptr<IRaycastFrustum> _frustum;

private:
    VolRenRaycast(); // Not implemented
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCAST_H
