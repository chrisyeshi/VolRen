#include "tfinteg2d.h"
#include "tfinteg2d.cuda.h"
#include <iostream>
#include <cassert>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>

TFInteg2D::TFInteg2D()
 : resolution(0)
{

}

TFInteg2D::~TFInteg2D()
{

}

QSharedPointer<QOpenGLTexture> TFInteg2D::newTexture(int size)
{
    this->resolution = size;
    auto tex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target2D));
    tex->setFormat(QOpenGLTexture::RGBA32F);
    tex->setSize(w(), h());
    tex->allocateStorage();
    return tex;
}

const std::unique_ptr<float[]>& TFInteg2D::integrate(QSharedPointer<QOpenGLTexture> tex, const float *colormap, float stepsize)
{
    assert(tex->width() == tex->height());
    this->tf.reset(new float [4 * w() * h()]);
    tfIntegrate2D(colormap, w(), stepsize, this->tf.get());
    tex->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, this->tf.get());
    return this->tf;
}

