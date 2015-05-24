#include "tfinteg1d.h"
#include <QOpenGLTexture>

namespace yy {
namespace volren {

TFInteg1D::TFInteg1D()
{

}

TFInteg1D::~TFInteg1D()
{

}

void TFInteg1D::integrate(const float *colormap, int resolution, float stepsize)
{
    if (!texture || resolution != texture->width())
    {
        texture.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
        texture->setFormat(QOpenGLTexture::RGBA32F);
        texture->setWrapMode(QOpenGLTexture::ClampToEdge);
        texture->setSize(resolution, 1);
        texture->allocateStorage();
        data.resize(resolution * 4);
    }
    const float baseSample = 0.01f;
    for (int i = 0; i < resolution; ++i)
    {
        // sample
        float spotR = colormap[4 * i + 0];
        float spotG = colormap[4 * i + 1];
        float spotB = colormap[4 * i + 2];
        float spotA = colormap[4 * i + 3];
        // adjust
        float adjustA = 1.f - std::pow(1.f - spotA, stepsize / baseSample);
        float adjustR = spotR * adjustA;
        float adjustG = spotG * adjustA;
        float adjustB = spotB * adjustA;
        // output
        data[4 * i + 0] = adjustR;
        data[4 * i + 1] = adjustG;
        data[4 * i + 2] = adjustB;
        data[4 * i + 3] = adjustA;
    }
    texture->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, data.data());
}

//QSharedPointer<QOpenGLTexture> TFInteg1D::newTexture(int size)
//{
//    this->resolution = size;
//    auto tex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target2D));
//    tex->setFormat(QOpenGLTexture::RGBA32F);
//    tex->setSize(w(), h());
//    tex->allocateStorage();
//    return tex;
//}

//const std::unique_ptr<float[]>& TFInteg1D::integrate(QSharedPointer<QOpenGLTexture> tex, const float *colormap, float stepsize)
//{
//    const float baseSample = 0.01f;
//    int size = tex->width();
//    this->tf.reset(new float [4 * size]);
//    for (int i = 0; i < tex->width(); ++i)
//    {
//        // sample
//        float spotR = colormap[4 * i + 0];
//        float spotG = colormap[4 * i + 1];
//        float spotB = colormap[4 * i + 2];
//        float spotA = colormap[4 * i + 3];
//        // adjust
//        float adjustA = 1.f - std::pow(1.f - spotA, stepsize / baseSample);
//        float adjustR = spotR * adjustA;
//        float adjustG = spotG * adjustA;
//        float adjustB = spotB * adjustA;
//        // output
//        this->tf[4 * i + 0] = adjustR;
//        this->tf[4 * i + 1] = adjustG;
//        this->tf[4 * i + 2] = adjustB;
//        this->tf[4 * i + 3] = adjustA;
//    }
//    tex->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, this->tf.get());
//    return this->tf;
//}

} // namespace volren
} // namespace yy

