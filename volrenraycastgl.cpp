#include "volrenraycastgl.h"
#include <map>
#include <sstream>
#include <cstdio>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include <volumegl.h>
#include "imagetex.h"

namespace yy {
namespace volren {

#define BASE VolRenRaycast

VolRenRaycastGL::VolRenRaycastGL()
 : BASE(Method_Raycast_GL)
{

}

std::unique_ptr<VolRenRaycastGL> VolRenRaycastGL::create()
{
    return std::unique_ptr<VolRenRaycastGL>(new VolRenRaycastGL());
}

VolRenRaycastGL::~VolRenRaycastGL()
{

}

void VolRenRaycastGL::initializeGL()
{
    // TODO: VolRenRaycast to TFIntegrated<VolRenRaycast> in VolRenRaycastCuda
    BASE::initializeGL();
    painter.initializeGL(":/volren/shaders/raycast.vert", ":/volren/shaders/raycast.frag");
    if (!colormap) colormap.reset(new ColormapGL());
    newFBOs();
}

void VolRenRaycastGL::resize(int w, int h)
{
    BASE::resize(w, h);
    newFBOs();
}

void VolRenRaycastGL::setVolume(const std::shared_ptr<IVolume> &volume)
{
    // check supported pixel type
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume->pixelType()))
    {
        std::cout << "Unsupported pixel type..." << std::endl;
        return;
    }
    // whether volume has the interface I requires
    std::shared_ptr<IVolumeGL> ptr = std::dynamic_pointer_cast<IVolumeGL>(volume);
    if (!ptr)
        ptr.reset(new VolumeGL(volume));
    this->volume = ptr;
    // set bounding box dimension
    _frustum.setVolSize(this->volume->w() * this->volume->sx(),
                        this->volume->h() * this->volume->sy(),
                        this->volume->d() * this->volume->sz());
}

void VolRenRaycastGL::setColormap(const std::shared_ptr<IColormap>& colormap)
{
    if (this->colormap == colormap)
        return;
    std::shared_ptr<IColormapGL> ptr = std::dynamic_pointer_cast<IColormapGL>(colormap);
    if (!ptr)
        ptr = std::make_shared<ColormapGL>(colormap);
    this->colormap = ptr;
}

std::shared_ptr<ImageAbstract> VolRenRaycastGL::output() const
{
    return std::make_shared<ImageTex>(*outTex);
}

void VolRenRaycastGL::newFBOs()
{
    newFBO(_frustum.texWidth(), _frustum.texHeight(), &outFBO, &outTex, &outRen);
}

void VolRenRaycastGL::newFBO(int w, int h, std::shared_ptr<GLuint> *fbo, std::shared_ptr<GLuint> *tex, std::shared_ptr<GLuint> *ren) const
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // texture
    tex->reset([](){
        GLuint* texPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenTextures(1, texPtr);
        return texPtr;
    }(), [](GLuint* texPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteTextures(1, texPtr);
    });
    f.glBindTexture(GL_TEXTURE_2D, **tex);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    // render buffer object
    ren->reset([](){
        GLuint* renPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenRenderbuffers(1, renPtr);
        return renPtr;
    }(), [](GLuint* renPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteRenderbuffers(1, renPtr);
    });
    f.glBindRenderbuffer(GL_RENDERBUFFER, **ren);
    f.glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
    f.glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // framebuffer object
    fbo->reset([](){
        GLuint* fboPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenFramebuffers(1, fboPtr);
        return fboPtr;
    }(), [](GLuint* fboPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteFramebuffers(1, fboPtr);
    });
    f.glBindFramebuffer(GL_FRAMEBUFFER, **fbo);
    f.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, **tex, 0);
    f.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, **ren);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);

    GLenum status;
    status = f.glCheckFramebufferStatus(GL_FRAMEBUFFER);
    switch (status)
    {
    case GL_FRAMEBUFFER_COMPLETE:
        break;
    default:
        std::cout << "framebuffer incomplete" << std::endl;
    }
}

void VolRenRaycastGL::raycast(const QMatrix4x4&, const QMatrix4x4& matView, const QMatrix4x4&)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glBindFramebuffer(GL_FRAMEBUFFER, *outFBO);
    GLint viewport[4];
    f.glGetIntegerv(GL_VIEWPORT, viewport);
    f.glViewport(0, 0, _frustum.texWidth(), _frustum.texHeight());
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, *_frustum.texEntry());
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, *_frustum.texExit());
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, volume->getTexture()->textureId());
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, colormap->texFull()->textureId());
    f.glActiveTexture(GL_TEXTURE4);
    f.glBindTexture(GL_TEXTURE_2D, colormap->texBack()->textureId());

    for (unsigned int i = 0; i < lights.size(); ++i)
    {
        QVector3D lightDir(matView.inverted() * QVector4D(lights[i].direction, 0.f));
        lightDir.normalize();
        painter.setUniforms(QString("lights[%1].direction").arg(i).toStdString().data(), lightDir,
                            QString("lights[%1].ambient").arg(i).toStdString().data(), lights[i].ambient * lights[i].color,
                            QString("lights[%1].diffuse").arg(i).toStdString().data(), lights[i].diffuse * lights[i].color,
                            QString("lights[%1].specular").arg(i).toStdString().data(), lights[i].specular * lights[i].color,
                            QString("lights[%1].shininess").arg(i).toStdString().data(), lights[i].shininess);
    }

    painter.paint("texEntry", 0,
                  "texExit", 1,
                  "texVolume", 2,
                  "texTFFull", 3,
                  "texTFBack", 4,
                  "volSize", QVector3D(volume->w(), volume->h(), volume->d()),
                  "stepSize", colormap->stepsize(),
                  "scalarMin", scalarMin,
                  "scalarMax", scalarMax,
                  "nLights", int(lights.size()));

    f.glActiveTexture(GL_TEXTURE4);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, 0);
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace volren
} // namespace yy
