#include "tfinteg1d.h"
#include <QOpenGLTexture>

TFInteg1D::TFInteg1D()
 : resolution(0)
{

}

TFInteg1D::~TFInteg1D()
{

}

QSharedPointer<QOpenGLTexture> TFInteg1D::newTexture(int size)
{
    this->resolution = size;
    auto tex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target2D));
    tex->setFormat(QOpenGLTexture::RGBA32F);
    tex->setSize(w(), h());
    tex->allocateStorage();
    return tex;
}

const std::unique_ptr<float[]>& TFInteg1D::integrate(QSharedPointer<QOpenGLTexture> tex, const float *colormap, float stepsize)
{
    const float baseSample = 0.01f;
    int size = tex->width();
    this->tf.reset(new float [4 * size]);
    for (int i = 0; i < tex->width(); ++i)
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
        this->tf[4 * i + 0] = adjustR;
        this->tf[4 * i + 1] = adjustG;
        this->tf[4 * i + 2] = adjustB;
        this->tf[4 * i + 3] = adjustA;
    }
    tex->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, this->tf.get());
    return this->tf;
}

