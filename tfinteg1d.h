#ifndef TFINTEG1D_H
#define TFINTEG1D_H

#include "tfintegrater.h"

class TFInteg1D : public TFIntegrater
{
public:
    TFInteg1D();
    virtual ~TFInteg1D();

// virtual functions from TFIntegrater
public:
    virtual QSharedPointer<QOpenGLTexture> newTexture(int size);
    virtual const std::unique_ptr<float[]>& integrate(QSharedPointer<QOpenGLTexture> tex, const float* colormap, float stepsize);
    virtual int w() const { return resolution; }
    virtual int h() const { return 1; }

private:
    int resolution;
};

#endif // TFINTEG1D_H
