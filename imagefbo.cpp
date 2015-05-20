#include "imagefbo.h"

using namespace yy;

ImageFBO::ImageFBO(const QSharedPointer<QOpenGLFramebufferObject> &fbo)
 : ImageAbstract(TYPE_FBO)
 , initialized(false)
 , fbo(fbo)
{

}

ImageFBO::~ImageFBO()
{

}

void ImageFBO::initialize()
{
    painter.initializeGL(":/volren/shaders/quad.vert", ":/volren/shaders/quad.frag");
    initialized = true;
}

void ImageFBO::draw()
{
    if (!initialized)
        initialize();
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, fbo->texture());
    painter.paint("tex", 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
