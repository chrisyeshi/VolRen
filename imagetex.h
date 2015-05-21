#ifndef IMAGETEX_H
#define IMAGETEX_H

#include "imageabstract.h"
#include <QtOpenGL>
#include "painter.h"

namespace yy {
namespace volren {

class ImageTex : public ImageAbstract
{
public:
    ImageTex(GLuint tex);
    ImageTex(const ImageTex& image);
    ImageTex &operator=(const ImageTex& image);
    virtual ~ImageTex();

    virtual void initialize();
    virtual void draw();

private:
    bool initialized;
    yy::Painter painter;
    GLuint tex;
};

} // namespace volren
} // namespace yy

#endif // IMAGETEX_H
