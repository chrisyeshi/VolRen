#ifndef VOLRENRAYCASTGL_H
#define VOLRENRAYCASTGL_H

#include "volrenraycast.h"
#include "tfintegrated.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "volume.h"
#include "painter.h"
#include "TF.h"
#include "tfintegrater.h"
#include "imageabstract.h"

namespace yy {

class IVolumeGL;

namespace volren {

class VolRenRaycastGL : public VolRenRaycast
{
public:
    VolRenRaycastGL();
    static std::unique_ptr<VolRenRaycastGL> create();
    virtual ~VolRenRaycastGL();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setVolume(const std::shared_ptr<IVolume> &volume);
    virtual void setColormap(const std::shared_ptr<IColormap> &colormap);
    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void raycast();

protected:
    void newFBOs();
    void newFBO(int w, int h, std::shared_ptr<GLuint>* fbo, std::shared_ptr<GLuint>* tex, std::shared_ptr<GLuint>* ren) const;

protected:
    Painter painter;
    std::shared_ptr<GLuint> outFBO, outTex, outRen;
    std::shared_ptr<IVolumeGL> volume;
    std::shared_ptr<IColormapGL> colormap;
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCASTGL_H
