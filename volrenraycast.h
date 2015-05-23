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

class IVolume;

namespace volren {

class VolumeGL;

class VolRenRaycast : public VolRen
{
public:
    VolRenRaycast(const Method& method);
    virtual ~VolRenRaycast();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setVolume(const std::weak_ptr<IVolume> &volume);
    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, Filter filter);
    virtual void render(const QMatrix4x4& v, const QMatrix4x4& p);
    virtual std::shared_ptr<ImageAbstract> output() const = 0;

protected:
    virtual void raycast(const QMatrix4x4& m, const QMatrix4x4& v, const QMatrix4x4& p) = 0;
    virtual void volumeChanged() {}
    virtual void tfChanged(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter) {}

protected:
    const int defaultFBOSize = 480;
    RaycastFurstum frustum;
    std::shared_ptr<VolumeGL> volume;
    QSharedPointer<QOpenGLTexture> tfTex;
    Filter tfFilter;
    bool preintegrate;
    std::unique_ptr<TFIntegrater> tfInteg;
    float stepsize;

protected:
    std::shared_ptr<VolumeGL> vol() const { return volume; }

private:
    VolRenRaycast(); // Not implemented
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCAST_H
