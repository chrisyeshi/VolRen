#ifndef TFINTEG1D_H
#define TFINTEG1D_H

#include "tfintegrater.h"
#include <QSharedPointer>
#include <QOpenGLTexture>

namespace yy {
namespace volren {

class TFInteg1D : public ITFIntegrater
{
public:
    TFInteg1D();
    virtual ~TFInteg1D();

public:
    virtual void integrate(const float *colormap, int resolution, float stepsize);
    virtual QSharedPointer<QOpenGLTexture> getTexture() const { return texture; }

private:
    QSharedPointer<QOpenGLTexture> texture;
    std::vector<float> data;
};

} // namespace volren
} // namespace yy

#endif // TFINTEG1D_H
