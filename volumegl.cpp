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

std::ostream& operator<<(std::ostream &os, const VolumeGL &volume)
{
    static std::map<Volume::ScalarType, std::string> dt2str
            = {{Volume::ST_Double, "Double"},
               {Volume::ST_Float, "Float"},
               {Volume::ST_Unsigned_Char, "Unsigned char"},
               {Volume::ST_Char, "Char"}};
    assert(0 != dt2str.count(volume.scalarType()));
    os << "Volume (type: " << dt2str[volume.scalarType()].c_str()
       << ") (size: [" << volume.w() << "," << volume.h() << "," << volume.d() << "])";
    return os;
}

void VolumeGL::makeTexture()
{
    static std::map<Volume::ScalarType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::ST_Char, QOpenGLTexture::Int8},
               {Volume::ST_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::ST_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume->scalarType()))
    {
        std::cout << "VolumeGL::Unsupported pixel type..." << std::endl;
        return;
    }
    texture = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target3D));
    texture->setFormat(QOpenGLTexture::R32F);
    texture->setSize(volume->w(), volume->h(), volume->d());
    texture->allocateStorage();
    if (1 == volume->nScalarsPerVoxel())
        texture->setData(QOpenGLTexture::Red, dt2pt[volume->scalarType()], volume->getData().get());
    else if (3 == volume->nScalarsPerVoxel())
        texture->setData(QOpenGLTexture::RGB, dt2pt[volume->scalarType()], volume->getData().get());
    else
        return;
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
