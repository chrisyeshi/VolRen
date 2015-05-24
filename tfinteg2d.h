#ifndef TFINTEG2D_H
#define TFINTEG2D_H

#include "tfintegrater.h"
#include <memory>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <QSharedPointer>
#include <QOpenGLTexture>

namespace yy {
namespace volren {

class TFInteg2D : public ITFIntegrater
{
public:
    TFInteg2D();
    virtual ~TFInteg2D();

public:
    virtual void integrate(const float* colormap, int resolution, float stepsize);
    virtual QSharedPointer<QOpenGLTexture> getTexture() const { return texture; }

private:
    QSharedPointer<QOpenGLTexture> texture;
    std::vector<float> data;

    void newResources(int resolution);
};

} // namespace volren
} // namespace yy

#endif // TFINTEG2D_H
