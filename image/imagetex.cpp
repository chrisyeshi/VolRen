#include "imagetex.h"
#include <QOpenGLFunctions>

namespace yy {
namespace volren {

ImageTex::ImageTex(GLuint tex)
 : ImageAbstract(TYPE_TEXTURE)
 , initialized(false)
 , tex(tex)
{

}

ImageTex::ImageTex(const ImageTex &image)
 : ImageAbstract(TYPE_TEXTURE)
 , initialized(false)
 , tex(image.tex)
{

}

ImageTex &ImageTex::operator=(const ImageTex &image)
{
    initialized = false;
    this->tex = image.tex;
    return *this;
}

ImageTex::~ImageTex()
{

}

void ImageTex::initialize()
{
    painter.initializeGL(":/volren/shaders/quad.vert", ":/volren/shaders/quad.frag");
    initialized = true;
}

void ImageTex::draw()
{
    if (!initialized)
        initialize();
    glBindTexture(GL_TEXTURE_2D, tex);
    painter.paint("tex", 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

std::vector<char> ImageTex::data() const
{
    auto f = QOpenGLContext::currentContext()->versionFunctions<QOpenGLFunctions_3_3_Core>();
    f->initializeOpenGLFunctions();
    f->glBindTexture(GL_TEXTURE_2D, tex);
    int width, height;
    f->glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    f->glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    std::vector<char> ret(width * height * 4);
    f->glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, ret.data());
    f->glBindTexture(GL_TEXTURE_2D, 0);
    return ret;
}

} // namespace volren
} // namespace yy
