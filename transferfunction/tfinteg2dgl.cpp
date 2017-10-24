#include "tfinteg2dgl.h"
#include <iostream>
#include <cassert>
#include <memory>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QOpenGLFunctions_3_3_Core>

namespace yy {
namespace volren {

TFInteg2DGL::TFInteg2DGL()
{

}

TFInteg2DGL::~TFInteg2DGL()
{

}

void TFInteg2DGL::integrate(const float *colormap, int resolution, float basesize, float stepsize)
{
    if (!texFull || resolution != texFull->width())
        newResources(resolution);
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    tex1d->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, colormap);
    // framebuffer object
    GLint oFbo, viewport[4], activeTex, oTex;
    newFbo();
    f->glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFbo);
    f->glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    f->glClear(GL_COLOR_BUFFER_BIT);
    // viewport
    f->glGetIntegerv(GL_VIEWPORT, viewport);
    f->glViewport(0, 0, resolution, resolution);
    // 1d texture
    f->glGetIntegerv(GL_ACTIVE_TEXTURE, &activeTex);
    f->glActiveTexture(GL_TEXTURE0);
    f->glGetIntegerv(GL_TEXTURE_1D, &oTex);
    f->glBindTexture(GL_TEXTURE_1D, tex1d->textureId());
    // paint
    painter.recreateVAO();
    painter.paint("tf1d", 0, "resolution", resolution, "basesize", basesize, "segLen", stepsize);
    // clean
    f->glActiveTexture(GL_TEXTURE0);
    f->glBindTexture(GL_TEXTURE_1D, oTex);
    f->glActiveTexture(activeTex);

//    std::unique_ptr<GLubyte[]> pixels(new GLubyte [resolution * resolution * 4]);
//    f->glReadPixels(0, 0, resolution, resolution, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
//    QImage image(pixels.get(), resolution, resolution, QImage::Format_RGBA8888);
//    static QLabel label;
//    label.resize(resolution, resolution);
//    label.setPixmap(QPixmap::fromImage(image.mirrored()));
//    label.show();

    f->glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f->glBindFramebuffer(GL_FRAMEBUFFER, oFbo);
}

void TFInteg2DGL::newResources(int resolution)
{
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    // painter
    painter.initializeGL(":/volren/shaders/quad.vert", ":/volren/shaders/transferfunction/tfinteg2d.frag");
    // new texture 1d
    tex1d.reset(new QOpenGLTexture(QOpenGLTexture::Target1D));
    tex1d->setFormat(QOpenGLTexture::RGBA32F);
    tex1d->setMinMagFilters(QOpenGLTexture::Nearest, QOpenGLTexture::Nearest);
    tex1d->setWrapMode(QOpenGLTexture::ClampToEdge);
    tex1d->setSize(resolution);
    tex1d->allocateStorage();
    // new texture full
    texFull.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
    texFull->setFormat(QOpenGLTexture::RGBA32F);
    texFull->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texFull->setWrapMode(QOpenGLTexture::ClampToEdge);
    texFull->setSize(resolution, resolution);
    texFull->allocateStorage();
    // new texture back
    texBack.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
    texBack->setFormat(QOpenGLTexture::RGBA32F);
    texBack->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texBack->setWrapMode(QOpenGLTexture::ClampToEdge);
    texBack->setSize(resolution, resolution);
    texBack->allocateStorage();
    // new fbo
    newFbo();
}

void TFInteg2DGL::newFbo()
{
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    // new fbo
    fbo.reset([](){
        GLuint* fboPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenFramebuffers(1, fboPtr);
        return fboPtr;
    }(), [](GLuint* fboPtr){
        QOpenGLContext::globalShareContext()->functions()->glDeleteFramebuffers(1, fboPtr);
    });
    GLint oFbo;
    f->glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFbo);
    f->glBindFramebuffer(GL_FRAMEBUFFER, *fbo);
    f->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texFull->textureId(), 0);
    f->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, texBack->textureId(), 0);
    GLenum bufs[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    f->glDrawBuffers(2, bufs);
    f->glBindFramebuffer(GL_FRAMEBUFFER, oFbo);
    GLenum status;
    status = f->glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        break;
    default:
        std::cout << "framebuffer back incomplete" << std::endl;
    }
}

} // namespace volren
} // namespace yy
