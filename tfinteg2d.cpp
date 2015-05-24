#include "tfinteg2d.h"
#include "tfinteg2d.cuda.h"
#include <iostream>
#include <cassert>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>

namespace yy {
namespace volren {

TFInteg2D::TFInteg2D()
{

}

TFInteg2D::~TFInteg2D()
{

}

void TFInteg2D::integrate(const float *colormap, int resolution, float stepsize)
{
    if (!texture || resolution != texture->width())
        newResources(resolution);
    tfIntegrate2D(colormap, resolution, stepsize, data.data());
    texture->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, data.data());
}

void TFInteg2D::newResources(int resolution)
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    // new texture
    texture.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
    texture->setFormat(QOpenGLTexture::RGBA32F);
    texture->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
    texture->setSize(resolution, resolution);
    texture->allocateStorage();
    // resize data
    data.resize(resolution * resolution * 4);
}

//QSharedPointer<QOpenGLTexture> TFInteg2D::newTexture(int size)
//{
//    this->resolution = size;
//    auto tex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target2D));
//    tex->setFormat(QOpenGLTexture::RGBA32F);
//    tex->setSize(w(), h());
//    tex->allocateStorage();
//    return tex;
//}

//const std::unique_ptr<float[]>& TFInteg2D::integrate(QSharedPointer<QOpenGLTexture> tex, const float *colormap, float stepsize)
//{
//    assert(tex->width() == tex->height());
//    this->tf.reset(new float [4 * w() * h()]);
//    tfIntegrate2D(colormap, w(), stepsize, this->tf.get());
//    tex->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, this->tf.get());
//    return this->tf;
//}

} // namespace volren
} // namespace yy

