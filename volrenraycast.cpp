#include "volrenraycast.h"
#include <map>
#include <cassert>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"
#include "volume.h"

using namespace yy;

VolRenRaycast::VolRenRaycast(const Method &method)
 : VolRen(method)
 , tfFilter(Filter_Linear)
 , preintegrate(false)
 , tfInteg(TFIntegrater::create(false))
 , stepsize(0.01f)
{

}

VolRenRaycast::~VolRenRaycast()
{
}

void VolRenRaycast::initializeGL()
{
    cube.initializeGL();
    painter.initializeGL(":/volren/shaders/raycast.vert", ":/volren/shaders/raycast.frag");
    newFBOs(defaultFBOSize, defaultFBOSize);
    this->setTF(mslib::TF(1024, 1024), preintegrate, stepsize, tfFilter);
}

void VolRenRaycast::resize(int w, int h)
{
    newFBOs(w, h);
}

void VolRenRaycast::setVolume(const std::weak_ptr<Volume> &volume)
{
    // if same volume
    if (this->volume.lock().get() == volume.lock().get())
        return;
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume.lock()->pixelType()))
    {
        std::cout << "Unsupported pixel type..." << std::endl;
        return;
    }
    this->volume = volume;
    // set cube dimension
    cube.setSize(vol()->w() * vol()->sx(), vol()->h() * vol()->sy(), vol()->d() * vol()->sz());
    // setup 3d texture
    volTex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target3D));
    volTex->setFormat(QOpenGLTexture::R32F);
    volTex->setSize(vol()->w(), vol()->h(), vol()->d());
    volTex->allocateStorage();
    volTex->setData(QOpenGLTexture::Red, dt2pt[vol()->pixelType()], vol()->getData().get());
    volTex->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    volTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    // volume changed
    volumeChanged();
}

void VolRenRaycast::setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter)
{
    if (this->preintegrate != preinteg)
    {
        this->preintegrate = preinteg;
        tfInteg = std::move(TFIntegrater::create(this->preintegrate));
        tfTex = tfInteg->newTexture(tf.resolution());

    } else if (tfTex.isNull() || tfTex->width() != tf.resolution())
    {
        tfTex = tfInteg->newTexture(tf.resolution());
    }
    tfFilter = filter;
    tfInteg->integrate(tfTex, tf.colorMap(), stepsize);
    static std::map<Filter, QOpenGLTexture::Filter> vr2qt
            = { { Filter_Linear, QOpenGLTexture::Linear }
              , { Filter_Nearest, QOpenGLTexture::Nearest } };
    assert(vr2qt.count(filter) > 0);
    tfTex->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
    tfTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    this->stepsize = stepsize;
    tfChanged(tf, preinteg, stepsize, filter);
}

void VolRenRaycast::render(const QMatrix4x4& v, const QMatrix4x4 &p)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // entry texture
    f.glBindFramebuffer(GL_FRAMEBUFFER, *entryFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    cube.render(p * v, GL_BACK);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // exit texture
    f.glBindFramebuffer(GL_FRAMEBUFFER, *exitFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    cube.render(p * v, GL_FRONT);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // raycast
    raycast(cube.matrix(), v, p);
}

void VolRenRaycast::newFBOs(int w, int h)
{
    newFBO(w, h, &entryFBO, &entryTex, &entryRen);
    newFBO(w, h, &exitFBO, &exitTex, &exitRen);
}

void VolRenRaycast::newFBO(int w, int h, std::shared_ptr<GLuint>* fbo, std::shared_ptr<GLuint>* tex, std::shared_ptr<GLuint>* ren) const
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

