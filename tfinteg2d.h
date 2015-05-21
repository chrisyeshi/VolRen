#ifndef TFINTEG2D_H
#define TFINTEG2D_H

#include "tfintegrater.h"

namespace yy {
namespace volren {

void kernel();

class TFInteg2D : public TFIntegrater
{
public:
    TFInteg2D();
    virtual ~TFInteg2D();

// virtual functions from TFIntegrater
public:
    virtual QSharedPointer<QOpenGLTexture> newTexture(int size);
    virtual const std::unique_ptr<float[]>& integrate(QSharedPointer<QOpenGLTexture> tex, const float* colormap, float stepsize);
    virtual int w() const { return resolution; }
    virtual int h() const { return resolution; }

private:
    int resolution;
};

} // namespace volren
} // namespace yy

#endif // TFINTEG2D_H
