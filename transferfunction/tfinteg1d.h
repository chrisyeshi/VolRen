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
    virtual void integrate(const float *colormap, int resolution, float basesize, float stepsize);
    virtual QSharedPointer<QOpenGLTexture> getTexFull() const { return texFull; }
    virtual QSharedPointer<QOpenGLTexture> getTexBack() const { return texBack; }

private:
    QSharedPointer<QOpenGLTexture> texFull, texBack;
    std::vector<float> data;
};

} // namespace volren
} // namespace yy

#endif // TFINTEG1D_H
