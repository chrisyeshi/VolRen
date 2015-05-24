#ifndef VOLRENRAYCASTGL_H
#define VOLRENRAYCASTGL_H

#include "volrenraycast.h"
#include "tfintegrated.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "volume.h"
#include "raycastcube.h"
#include "painter.h"
#include "TF.h"
#include "tfintegrater.h"
#include "imageabstract.h"

namespace yy {
namespace volren {

class VolRenRaycastGL : public TFIntegrated<VolRenRaycast>
{
public:
    VolRenRaycastGL();
    virtual ~VolRenRaycastGL();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&);

private:
    void newFBOs();
    void newFBO(int w, int h, std::shared_ptr<GLuint>* fbo, std::shared_ptr<GLuint>* tex, std::shared_ptr<GLuint>* ren) const;

protected:
    Painter painter;
    std::shared_ptr<GLuint> outFBO, outTex, outRen;
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCASTGL_H
