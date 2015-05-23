#include "raycastfurstum.h"

#include <QOpenGLFunctions>

namespace yy {
namespace volren {

RaycastFurstum::RaycastFurstum()
 : texWidth(defaultFBOSize), texHeight(defaultFBOSize)
 , volWidth(0), volHeight(0), volDepth(0)
{

}

RaycastFurstum::~RaycastFurstum()
{

}

void RaycastFurstum::initializeGL()
{
    cube.initializeGL();
    newFBOs();
    // the plane shape
    plane = Shape::quad();
    Shape::Attribute attr;
    std::vector<GLfloat> texCoords(4 * 3);
    attr.set(GL_FLOAT, 3, 0, texCoords);
    plane.attributes.push_back(attr);
    // initialize near and far painter
    pNear.initializeGL(plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplane.frag");
    pFar.initializeGL(plane, ":/volren/shaders/raycastplane.vert", ":/volren/shaders/raycastplane.frag");
}

void RaycastFurstum::setResolution(int w, int h)
{
    texWidth = w;
    texHeight = h;
    newFBOs();
}

void RaycastFurstum::setVolumeDimension(int w, int h, int d)
{
    volWidth = w;
    volHeight = h;
    volDepth = d;
    cube.setSize(w, h, d);
}

std::shared_ptr<GLuint> RaycastFurstum::entryTexture(const QMatrix4x4 &v, const QMatrix4x4 &p)
{
    // near plane geometry
    QMatrix4x4 imvp = (p * v * modelMatrix()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, -1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, -1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, -1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, -1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    plane.attributes[1].setData(attr);
    pNear.updateAttributes(plane.attributes);

    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glBindFramebuffer(GL_FRAMEBUFFER, *entryFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pNear.paint("volDim", QVector3D(volWidth, volHeight, volDepth));
    cube.render(p * v, GL_BACK);

//    std::unique_ptr<GLubyte[]> pixels(new GLubyte [texWidth * texHeight * 3]);
//    f.glReadPixels(0, 0, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.get());
//    QImage image(pixels.get(), texWidth, texHeight, QImage::Format_RGB888);
//    static QLabel label;
//    label.resize(texWidth, texHeight);
//    label.setPixmap(QPixmap::fromImage(image));
//    label.show();

    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return entryTex;
}

std::shared_ptr<GLuint> RaycastFurstum::exitTexture(const QMatrix4x4 &v, const QMatrix4x4 &p)
{
    // far plane geometry
    QMatrix4x4 imvp = (p * v * modelMatrix()).inverted();
    QVector3D p0 = imvp * QVector3D(-1.f, -1.f, 1.f);
    QVector3D p1 = imvp * QVector3D( 1.f, -1.f, 1.f);
    QVector3D p2 = imvp * QVector3D( 1.f,  1.f, 1.f);
    QVector3D p3 = imvp * QVector3D(-1.f,  1.f, 1.f);
    std::vector<GLfloat> attr = {p0.x(), p0.y(), p0.z(),
                                 p1.x(), p1.y(), p1.z(),
                                 p2.x(), p2.y(), p2.z(),
                                 p3.x(), p3.y(), p3.z()};
    plane.attributes[1].setData(attr);
    pFar.updateAttributes(plane.attributes);

    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glBindFramebuffer(GL_FRAMEBUFFER, *exitFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    pFar.paint("volDim", QVector3D(volWidth, volHeight, volDepth));
    cube.render(p * v, GL_FRONT);

//    std::unique_ptr<GLubyte[]> pixels(new GLubyte [texWidth * texHeight * 3]);
//    f.glReadPixels(0, 0, texWidth, texHeight, GL_RGB, GL_UNSIGNED_BYTE, pixels.get());
//    QImage image(pixels.get(), texWidth, texHeight, QImage::Format_RGB888);
//    static QLabel label;
//    label.resize(texWidth, texHeight);
//    label.setPixmap(QPixmap::fromImage(image));
//    label.show();

    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return exitTex;
}

void RaycastFurstum::newFBOs()
{
    newFBO(texWidth, texHeight, &entryFBO, &entryTex, &entryRen);
    newFBO(texWidth, texHeight,  &exitFBO,  &exitTex,  &exitRen);
}

void RaycastFurstum::newFBO(int w, int h, std::shared_ptr<GLuint> *fbo, std::shared_ptr<GLuint> *tex, std::shared_ptr<GLuint> *ren) const
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

} // namespace volren
} // namespace yy
