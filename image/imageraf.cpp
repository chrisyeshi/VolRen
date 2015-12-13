#include "imageraf.h"
#include <QOpenGLFunctions_3_3_Core>

namespace yy {
namespace volren {

ImageRAF::ImageRAF(std::shared_ptr<GLuint> rafPBO, std::shared_ptr<GLuint> depPBO, int w, int h, int layers, QSharedPointer<QOpenGLTexture> tf)
 : ImageAbstract(TYPE_PBO_RAF)
 , initialized(false)
 , rafPBO(rafPBO)
 , depPBO(depPBO)
 , tf(tf)
 , w(w), h(h)
 , layers(layers)
{

}

ImageRAF::~ImageRAF()
{

}

void ImageRAF::initialize()
{
    painter.initializeGL(":/volren/shaders/image/imageraf.vert", ":/volren/shaders/image/imageraf.frag");
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    tex.reset([](){
        GLuint* texPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenTextures(1, texPtr);
        return texPtr;
    }(), [](GLuint* texPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteTextures(1, texPtr);
    });
    f->glBindTexture(GL_TEXTURE_3D, *tex);
    f->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    f->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    f->glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    f->glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, w, h, layers, 0, GL_RED, GL_FLOAT, NULL);
    f->glBindTexture(GL_TEXTURE_3D, 0);
    initialized = true;
}

void ImageRAF::draw()
{
    if (!initialized)
        this->initialize();
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_1D);
    glEnable(GL_TEXTURE_3D);
    f->glActiveTexture(GL_TEXTURE0);
    f->glBindTexture(GL_TEXTURE_1D, tf->textureId());
    f->glActiveTexture(GL_TEXTURE1);
    f->glBindTexture(GL_TEXTURE_3D, *tex);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *rafPBO);
    f->glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, w, h, layers, GL_RED, GL_FLOAT, 0);
    painter.paint("texTF", 0, "texRAF", 1, "layers", layers);
    f->glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    f->glActiveTexture(GL_TEXTURE1);
    f->glBindTexture(GL_TEXTURE_3D, 0);
    f->glActiveTexture(GL_TEXTURE0);
    f->glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_3D);
    glDisable(GL_TEXTURE_1D);
}

} // namespace volren
} // namespace yy
