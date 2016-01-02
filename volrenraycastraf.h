#ifndef VOLRENRAYCASTRAF_H
#define VOLRENRAYCASTRAF_H

#include "volrenraycast.h"
#include <QtOpenGL>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "TF.h"

namespace yy {
namespace volren {

class VolRenRaycastRAF : public VolRenRaycast
{
public:
    VolRenRaycastRAF();
    static std::unique_ptr<VolRenRaycastRAF> create();
    virtual ~VolRenRaycastRAF();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter);
    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void newFBOs(int w, int h);
    virtual void raycast(const QMatrix4x4& m, const QMatrix4x4& v, const QMatrix4x4& p);
    virtual void volumeChanged();

protected:
    static const int defaultLayers = 8;
    static const int maxLayers = 8;
    std::shared_ptr<GLuint> rafPBO, depPBO;
    cudaGraphicsResource *entryRes, *exitRes, *rafRes, *depRes;
    cudaGraphicsResource *volRes;
    std::shared_ptr<cudaArray> tfArr;
    int texWidth, texHeight, layers;
    bool preintegrate;
    Filter tfFilter;
    float stepsize;
    QSharedPointer<QOpenGLTexture> tfTex;

private:
    void newOutPBO(std::shared_ptr<GLuint> *outPBO, cudaGraphicsResource **outRes, int w, int h, int l);
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCASTRAF_H
