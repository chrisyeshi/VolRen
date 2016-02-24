#include "imagepbo.h"
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_3_3_Core>

namespace yy {
namespace volren {

ImagePBO::ImagePBO(GLuint pbo, int w, int h)
 : ImageAbstract(TYPE_PBO)
 , initialized(false)
 , pbo(pbo)
 , tex(0)
 , w(w), h(h)
{
}

ImagePBO::~ImagePBO()
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    if (0 != tex) f.glDeleteTextures(1, &tex);
}

void ImagePBO::initialize()
{
    painter.initializeGL(":/volren/shaders/quad.vert", ":/volren/shaders/quad.frag");
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glGenTextures(1, &tex);
    f.glBindTexture(GL_TEXTURE_2D, tex);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    initialized = true;
}

void ImagePBO::draw()
{
    if (!initialized)
        initialize();
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
//    f->glClear(GL_COLOR_BUFFER_BIT);
    f->glBindTexture(GL_TEXTURE_2D, tex);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    f->glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, 0);
    painter.paint("tex", 0);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    f->glBindTexture(GL_TEXTURE_2D, 0);
}

} // namespace volren
} // namespace yy
