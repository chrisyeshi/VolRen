#ifndef VOLRENRAYCAST_H
#define VOLRENRAYCAST_H

#include "volren.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "raycastfurstum.h"
#include "raycastcube.h"
#include "TF.h"
#include "tfintegrater.h"
#include "imageabstract.h"

namespace yy {
namespace volren {

class VolRenRaycast : public VolRen
{
public:
    VolRenRaycast(const Method& method);
    virtual ~VolRenRaycast();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    // virtual void setVolume(const std::shared_ptr<IVolume> &volume);
    virtual void render(const QMatrix4x4& v, const QMatrix4x4& p);
    virtual std::shared_ptr<ImageAbstract> output() const = 0;

    virtual void setParaSheet(const Json::Value &json);
    virtual Json::Value getParaSheet() const;

protected:
    virtual void raycast(const QMatrix4x4& m, const QMatrix4x4& v, const QMatrix4x4& p) = 0;

protected:
    RaycastFurstum frustum;
    // std::shared_ptr<VolumeGL> volume;

protected:
    // std::shared_ptr<VolumeGL> vol() const { return volume; }

private:
    VolRenRaycast(); // Not implemented
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCAST_H
