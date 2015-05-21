#ifndef IMAGEFBO_H
#define IMAGEFBO_H

#include "imageabstract.h"
#include <QSharedPointer>
#include <QOpenGLFramebufferObject>
#include "painter.h"

namespace yy {
namespace volren {

class ImageFBO : public ImageAbstract
{
public:
    ImageFBO(const QSharedPointer<QOpenGLFramebufferObject>& fbo);
    virtual ~ImageFBO();

    virtual void initialize();
    virtual void draw();

private:
    bool initialized;
    yy::Painter painter;
    QSharedPointer<QOpenGLFramebufferObject> fbo;
};

} // namespace volren
} // namespace yy

#endif // IMAGEFBO_H
