#include "raycastfrustum.h"

#include <QOpenGLFunctions>

namespace yy {
namespace volren {

RaycastFrustum::RaycastFrustum()
  : _texWidth(_defaultFBOSize), _texHeight(_defaultFBOSize)
  , _volWidth(0), _volHeight(0), _volDepth(0)
  , _isFBOUpdated(false), _isGLInitialized(false)
  , _isEntryUpdated(false), _isExitUpdated(false)
{

}

RaycastFrustum::~RaycastFrustum()
{

}

QMatrix4x4 RaycastFrustum::matModel() const
{
    static QMatrix4x4 mat;
    mat.setToIdentity();
    mat.scale(_volWidth, _volHeight, _volDepth);
    return mat;
}

void RaycastFrustum::setTexSize(int w, int h)
{
    _texWidth = w;
    _texHeight = h;
    _isFBOUpdated = false;
    _isEntryUpdated = false;
    _isExitUpdated = false;
}

void RaycastFrustum::setVolSize(int w, int h, int d)
{
    _volWidth = w;
    _volHeight = h;
    _volDepth = d;
    _isEntryUpdated = false;
    _isExitUpdated = false;
}

void RaycastFrustum::setMatView(const QMatrix4x4 &matView)
{
    _matView = matView;
    _isEntryUpdated = false;
    _isExitUpdated = false;
}

void RaycastFrustum::setMatProj(const QMatrix4x4 &matProj)
{
    _matProj = matProj;
    _isEntryUpdated = false;
    _isExitUpdated = false;
}

std::shared_ptr<GLuint> RaycastFrustum::texEntry() const
{
    if (!_isGLInitialized)
        initializeGL();
    if (!_isFBOUpdated)
        newFBOs();
    if (!_isEntryUpdated)
        makeEntry();
    return _entryTex;
}

std::shared_ptr<GLuint> RaycastFrustum::texExit() const
{
    if (!_isGLInitialized)
        initializeGL();
    if (!_isFBOUpdated)
        newFBOs();
    if (!_isExitUpdated)
        makeExit();
    return _exitTex;
}

void RaycastFrustum::initializeGL() const
{
    _plane = Shape::quad();
    Shape::Attribute attr;
    std::vector<GLfloat> texCoords(4 * 3);
    attr.set(GL_FLOAT, 3, 0, texCoords);
    _plane.attributes.push_back(attr);
    _pCube.reset(new Painter());
    _pCube->initializeGL(yy::Shape::cube(), ":/volren/shaders/raycastcube.vert", ":/volren/shaders/raycastcube.frag");
    _pNear.reset(new Painter());
    _pNear->initializeGL(_plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplane.frag");
    _pFar.reset(new Painter());
    _pFar->initializeGL(_plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplane.frag");
    _isGLInitialized = true;
}

void RaycastFrustum::newFBOs() const
{
    newFBO(_texWidth, _texHeight, &_entryFBO, &_entryTex, &_entryRen);
    newFBO(_texWidth, _texHeight,  &_exitFBO,  &_exitTex,  &_exitRen);
    _isFBOUpdated = true;
}

void RaycastFrustum::newFBO(int w, int h, std::shared_ptr<GLuint> *fbo, std::shared_ptr<GLuint> *tex, std::shared_ptr<GLuint> *ren) const
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    GLint oTex, oFbo, oRen;
    // texture
    tex->reset([](){
        GLuint* texPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenTextures(1, texPtr);
        return texPtr;
    }(), [](GLuint* texPtr){
        QOpenGLContext::globalShareContext()->functions()->glDeleteTextures(1, texPtr);
    });
    f.glGetIntegerv(GL_TEXTURE_BINDING_2D, &oTex);
    f.glBindTexture(GL_TEXTURE_2D, **tex);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
    f.glBindTexture(GL_TEXTURE_2D, oTex);
    // render buffer object
    ren->reset([](){
        GLuint* renPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenRenderbuffers(1, renPtr);
        return renPtr;
    }(), [](GLuint* renPtr){
        QOpenGLContext::globalShareContext()->functions()->glDeleteRenderbuffers(1, renPtr);
    });
    f.glGetIntegerv(GL_RENDERBUFFER_BINDING, &oRen);
    f.glBindRenderbuffer(GL_RENDERBUFFER, **ren);
    f.glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
    f.glBindRenderbuffer(GL_RENDERBUFFER, oRen);
    // framebuffer object
    fbo->reset([](){
        GLuint* fboPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenFramebuffers(1, fboPtr);
        return fboPtr;
    }(), [](GLuint* fboPtr){
        QOpenGLContext::globalShareContext()->functions()->glDeleteFramebuffers(1, fboPtr);
    });
    f.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFbo);
    f.glBindFramebuffer(GL_FRAMEBUFFER, **fbo);
    f.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, **tex, 0);
    f.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, **ren);
    f.glBindFramebuffer(GL_FRAMEBUFFER, oFbo);

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

void RaycastFrustum::makeEntry() const
{
    // near plane geometry
    QMatrix4x4 imvp = (matProj() * matView() * matModel()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, -1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, -1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, -1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, -1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    _plane.attributes[1].setData(attr);
    _pNear->updateAttributes(_plane.attributes);
    // setup GL states
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    GLint oFBO, viewport[4];
    f.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFBO);
    f.glBindFramebuffer(GL_FRAMEBUFFER, *_entryFBO);
    f.glGetIntegerv(GL_VIEWPORT, viewport);
    f.glViewport(0, 0, _texWidth, _texHeight);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // paint the entry texture
    _pNear->paint("volDim", QVector3D(_volWidth, _volHeight, _volDepth));
    // paint the cube
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    _pCube->paint("mvp", matProj() * matView() * matModel(),
                  "volDim", QVector3D(_volWidth, _volHeight, _volDepth));
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    // reset GL states
    f.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f.glBindFramebuffer(GL_FRAMEBUFFER, oFBO);
    _isEntryUpdated = true;
}

void RaycastFrustum::makeExit() const
{
    // far plane geometry
    QMatrix4x4 imvp = (matProj() * matView() * matModel()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, 1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, 1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, 1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, 1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    _plane.attributes[1].setData(attr);
    _pFar->updateAttributes(_plane.attributes);
    // setup GL states
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    GLint oFBO, viewport[4];
    f.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFBO);
    f.glBindFramebuffer(GL_FRAMEBUFFER, *_exitFBO);
    f.glGetIntegerv(GL_VIEWPORT, viewport);
    f.glViewport(0, 0, _texWidth, _texHeight);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // paint the entry texture
    _pFar->paint("volDim", QVector3D(_volWidth, _volHeight, _volDepth));
    // paint the cube
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    _pCube->paint("mvp", matProj() * matView() * matModel(),
                  "volDim", QVector3D(_volWidth, _volHeight, _volDepth));
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    // reset GL states
    f.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f.glBindFramebuffer(GL_FRAMEBUFFER, oFBO);
    _isExitUpdated = true;
}

FrustumProgressive::FrustumProgressive(int level)
  : _progress(0)
  , _pgrLevel(level)
{

}

FrustumProgressive::~FrustumProgressive()
{

}

void FrustumProgressive::setProgressLevel(int level)
{
    _pgrLevel = level;
    resetProgress();
}

void FrustumProgressive::resetProgress()
{
    _progress = 0;
    _isEntryUpdated = false;
    _isExitUpdated = false;
}

bool FrustumProgressive::advance()
{
    ++_progress;
    _isEntryUpdated = false;
    _isExitUpdated = false;
    return finished();
}

bool FrustumProgressive::finished() const
{
    if (_progress >= nProgress())
        return true;
    return false;
}

void FrustumProgressive::initializeGL() const
{
    _plane = Shape::quad();
    Shape::Attribute attr;
    std::vector<GLfloat> texCoords(4 * 3);
    attr.set(GL_FLOAT, 3, 0, texCoords);
    _plane.attributes.push_back(attr);
    _pCube.reset(new Painter());
    _pCube->initializeGL(yy::Shape::cube(), ":/volren/shaders/raycastcube.vert", ":/volren/shaders/raycastcubeprogressive.frag");
    _pNear.reset(new Painter());
    _pNear->initializeGL(_plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplaneprogressive.frag");
    _pFar.reset(new Painter());
    _pFar->initializeGL(_plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplaneprogressive.frag");
    _isGLInitialized = true;
}

void FrustumProgressive::makeEntry() const
{
    // near plane geometry
    QMatrix4x4 imvp = (matProj() * matView() * matModel()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, -1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, -1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, -1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, -1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    _plane.attributes[1].setData(attr);
    _pNear->updateAttributes(_plane.attributes);
    // setup GL states
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    GLint oFBO, viewport[4];
    f.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFBO);
    f.glBindFramebuffer(GL_FRAMEBUFFER, *_entryFBO);
    f.glGetIntegerv(GL_VIEWPORT, viewport);
    f.glViewport(0, 0, _texWidth, _texHeight);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // paint the entry texture
    _pNear->paint("volDim", QVector3D(_volWidth, _volHeight, _volDepth),
                  "progress", _progress,
                  "pgrLevel", _pgrLevel);
    // paint the cube
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    _pCube->paint("mvp", matProj() * matView() * matModel(),
                  "volDim", QVector3D(_volWidth, _volHeight, _volDepth),
                  "progress", _progress,
                  "pgrLevel", _pgrLevel);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

//    std::unique_ptr<GLubyte[]> pixels(new GLubyte [_texWidth * _texHeight * 4]);
//    glReadPixels(0, 0, _texWidth, _texHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
//    QImage image(pixels.get(), _texWidth, _texHeight, QImage::Format_RGBA8888);
//    static QLabel label;
//    label.resize(_texWidth, _texHeight);
//    label.setPixmap(QPixmap::fromImage(image.mirrored()));
//    label.show();

    // reset GL states
    f.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f.glBindFramebuffer(GL_FRAMEBUFFER, oFBO);
    _isEntryUpdated = true;
}

void FrustumProgressive::makeExit() const
{
    // far plane geometry
    QMatrix4x4 imvp = (matProj() * matView() * matModel()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, 1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, 1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, 1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, 1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    _plane.attributes[1].setData(attr);
    _pFar->updateAttributes(_plane.attributes);
    // setup GL states
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    GLint oFBO, viewport[4];
    f.glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oFBO);
    f.glBindFramebuffer(GL_FRAMEBUFFER, *_exitFBO);
    f.glGetIntegerv(GL_VIEWPORT, viewport);
    f.glViewport(0, 0, _texWidth, _texHeight);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // paint the entry texture
    _pFar->paint("volDim", QVector3D(_volWidth, _volHeight, _volDepth),
                 "progress", _progress,
                 "pgrLevel", _pgrLevel);
    // paint the cube
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_FRONT);
    _pCube->paint("mvp", matProj() * matView() * matModel(),
                  "volDim", QVector3D(_volWidth, _volHeight, _volDepth),
                  "progress", _progress,
                  "pgrLevel", _pgrLevel);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);

//    std::unique_ptr<GLubyte[]> pixels(new GLubyte [_texWidth * _texHeight * 4]);
//    glReadPixels(0, 0, _texWidth, _texHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels.get());
//    QImage image(pixels.get(), _texWidth, _texHeight, QImage::Format_RGBA8888);
//    static QLabel label;
//    label.resize(_texWidth, _texHeight);
//    label.setPixmap(QPixmap::fromImage(image.mirrored()));
//    label.show();

    // reset GL states
    f.glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);
    f.glBindFramebuffer(GL_FRAMEBUFFER, oFBO);
    _isExitUpdated = true;
}

} // namespace volren
} // namespace yy
