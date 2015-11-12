#include "volumegl.h"
#include <cassert>

namespace yy {
namespace volren {

std::map<Filter, QOpenGLTexture::Filter> VolumeGL::filter2qgl
        = { { Filter_Linear, QOpenGLTexture::Linear },
            { Filter_Nearest, QOpenGLTexture::Nearest } };

VolumeGL::~VolumeGL()
{

}

void VolumeGL::makeTexture()
{
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume->pixelType()))
    {
        std::cout << "VolumeGL::Unsupported pixel type..." << std::endl;
        return;
    }
    texture = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target3D));
    texture->setFormat(QOpenGLTexture::R32F);
    texture->setSize(volume->w(), volume->h(), volume->d());
    texture->allocateStorage();
    texture->setData(QOpenGLTexture::Red, dt2pt[volume->pixelType()], volume->getData().get());
    assert(filter2qgl.count(filter) > 0);
    texture->setMinMagFilters(filter2qgl[filter], filter2qgl[filter]);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
}

void VolumeGL::setFilter(Filter filter)
{
    this->filter = filter;
    if (texture)
    {
        assert(filter2qgl.count(filter) > 0);
        texture->setMinMagFilters(filter2qgl[filter], filter2qgl[filter]);
    }
}

} // namespace volren
} // namespace yy
