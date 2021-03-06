#ifndef IMAGEPBO_H
#define IMAGEPBO_H

#include "imageabstract.h"
#include <QtOpenGL>
#include "painter.h"

namespace yy {
namespace volren {

class ImagePBO : public ImageAbstract
{
public:
    ImagePBO(GLuint pbo, int w, int h);
    virtual ~ImagePBO();

    virtual void initialize();
    virtual void draw();

private:
    bool initialized;
    yy::Painter painter;
    GLuint pbo, tex;
    int w, h;
};

} // namespace volren
} // namespace yy

#endif // IMAGEPBO_H
