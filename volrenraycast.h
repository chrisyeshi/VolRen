#ifndef VOLRENRAYCAST_H
#define VOLRENRAYCAST_H

#include "volren.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "raycastcube.h"
#include "painter.h"
#include "TF.h"
#include "tfintegrater.h"
#include "imageabstract.h"

namespace yy {
class Volume;
}

class VolRenRaycast : public VolRen
{
public:
    VolRenRaycast(const Method& method);
    virtual ~VolRenRaycast();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setVolume(const std::weak_ptr<yy::Volume>& volume);
    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, VolRen::Filter filter);
    virtual void render(const QMatrix4x4& v, const QMatrix4x4& p);
    virtual std::shared_ptr<ImageAbstract> output() const = 0;

protected:
    virtual void newFBOs(int w, int h);
    virtual void raycast(const QMatrix4x4& m, const QMatrix4x4& v, const QMatrix4x4& p) = 0;
    virtual void volumeChanged() {}
    virtual void tfChanged(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter) {}
    void newFBO(int w, int h, std::shared_ptr<GLuint>* fbo, std::shared_ptr<GLuint>* tex, std::shared_ptr<GLuint>* ren) const;

protected:
    const int defaultFBOSize = 480;
    RaycastCube cube;
    yy::Painter painter;
    std::shared_ptr<GLuint> entryFBO, exitFBO;
    std::shared_ptr<GLuint> entryTex, exitTex;
    std::shared_ptr<GLuint> entryRen, exitRen;
    std::weak_ptr<yy::Volume> volume;
    QSharedPointer<QOpenGLTexture> volTex;
    QSharedPointer<QOpenGLTexture> tfTex;
    Filter tfFilter;
    bool preintegrate;
    std::unique_ptr<TFIntegrater> tfInteg;
    float stepsize;

protected:
    std::shared_ptr<yy::Volume> vol() const { return volume.lock(); }

private:
    VolRenRaycast(); // Not implemented
};

#endif // VOLRENRAYCAST_H
