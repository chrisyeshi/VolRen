#include "volumegl.h"

namespace yy {
namespace volren {

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
    texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
}

} // namespace volren
} // namespace yy
