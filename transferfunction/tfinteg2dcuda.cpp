#include "tfinteg2dcuda.h"
#include "tfinteg2d.cuda.h"
#include <iostream>
#include <cassert>
#include <QOpenGLTexture>
#include <QOpenGLFunctions>
#include <QLabel>
#include <QImage>

namespace yy {
namespace volren {

TFInteg2DCUDA::TFInteg2DCUDA()
{

}

TFInteg2DCUDA::~TFInteg2DCUDA()
{

}

void TFInteg2DCUDA::integrate(const float *colormap, int resolution, float basesize, float stepsize)
{
    if (!texFull || resolution != texFull->width())
        newResources(resolution);
    tfIntegrate2D(colormap, resolution, basesize, stepsize, dataFull.data(), dataBack.data());
    texFull->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, dataFull.data());
    texBack->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, dataBack.data());
}

void TFInteg2DCUDA::newResources(int resolution)
{
    // new texture full
    texFull.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
    texFull->setFormat(QOpenGLTexture::RGBA32F);
    texFull->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texFull->setWrapMode(QOpenGLTexture::ClampToEdge);
    texFull->setSize(resolution, resolution);
    texFull->allocateStorage();
    // new texture back
    texBack.reset(new QOpenGLTexture(QOpenGLTexture::Target2D));
    texBack->setFormat(QOpenGLTexture::RGBA32F);
    texBack->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    texBack->setWrapMode(QOpenGLTexture::ClampToEdge);
    texBack->setSize(resolution, resolution);
    texBack->allocateStorage();
    // resize data
    dataFull.resize(resolution * resolution * 4);
    dataBack.resize(resolution * resolution * 4);
}

} // namespace volren
} // namespace yy

