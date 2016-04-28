#ifndef TFINTEG2DCUDA_H
#define TFINTEG2DCUDA_H

#include "tfintegrater.h"
#include <memory>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QSharedPointer>
#include <QOpenGLTexture>

namespace yy {
namespace volren {

class TFInteg2DCUDA : public ITFIntegrater
{
public:
    TFInteg2DCUDA();
    virtual ~TFInteg2DCUDA();

public:
    virtual void integrate(const float* colormap, int resolution, float basesize, float stepsize);
    virtual QSharedPointer<QOpenGLTexture> getTexFull() const { return texFull; }
    virtual QSharedPointer<QOpenGLTexture> getTexBack() const { return texBack; }

private:
    QSharedPointer<QOpenGLTexture> texFull, texBack;
    std::vector<float> dataFull, dataBack;

    void newResources(int resolution);
};

} // namespace volren
} // namespace yy

#endif // TFINTEG2DCUDA_H
