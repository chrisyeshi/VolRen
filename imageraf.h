#ifndef IMAGERAF_H
#define IMAGERAF_H

#include "imageabstract.h"
#include <memory>
#include <QtOpenGL>
#include <QOpenGLTexture>
#include <QSharedPointer>
#include "painter.h"

namespace yy {
namespace volren {

class ImageRAF : public ImageAbstract
{
    friend class ImageWriterRAF;

public:
    ImageRAF(std::shared_ptr<GLuint> rafPBO, std::shared_ptr<GLuint> depPBO,
             int w, int h, int layers, QSharedPointer<QOpenGLTexture> tf);
    virtual ~ImageRAF();

    virtual void initialize();
    virtual void draw();

private:
    bool initialized;
    yy::Painter painter;
    std::shared_ptr<GLuint> rafPBO, depPBO, tex;
    QSharedPointer<QOpenGLTexture> tf;
    int w, h, layers;
};

} // namespace volren
} // namespace yy

#endif // IMAGERAF_H
