#include "volrenraycastgl.h"
#include <map>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"

namespace yy {
namespace volren {

VolRenRaycastGL::VolRenRaycastGL()
 : VolRenRaycast(Method_Raycast_GL)
{

}

VolRenRaycastGL::~VolRenRaycastGL()
{

}

void VolRenRaycastGL::initializeGL()
{
    VolRenRaycast::initializeGL();
    painter.initializeGL(":/volren/shaders/raycast.vert", ":/volren/shaders/raycast.frag");
    newFBOs();
}

void VolRenRaycastGL::resize(int w, int h)
{
    VolRenRaycast::resize(w, h);
    newFBOs();
}

std::shared_ptr<ImageAbstract> VolRenRaycastGL::output() const
{
    return std::make_shared<ImageTex>(*outTex);
}

void VolRenRaycastGL::newFBOs()
{
    newFBO(frustum.getTextureWidth(), frustum.getTextureHeight(), &outFBO, &outTex, &outRen);
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

void VolRenRaycastGL::raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glBindFramebuffer(GL_FRAMEBUFFER, *outFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f.glEnable(GL_TEXTURE_2D);
    f.glEnable(GL_TEXTURE_3D);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, *frustum.entryTexture());
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, *frustum.exitTexture());
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, volTex->textureId());
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, tfTex->textureId());
    painter.paint("texEntry", 0,
                  "texExit", 1,
                  "texVolume", 2,
                  "texTF", 3,
                  "volSize", QVector3D(vol()->w(), vol()->h(), vol()->d()),
                  "stepSize", stepsize,
                  "scalarMin", scalarMin,
                  "scalarMax", scalarMax);
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, 0);
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glDisable(GL_TEXTURE_3D);
    f.glDisable(GL_TEXTURE_2D);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace volren
} // namespace yy
