#ifndef VOLRENRAYCASTRAF_H
#define VOLRENRAYCASTRAF_H

#include "volrenraycast.h"
#include <QtOpenGL>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <colormap.h>
#include "TF.h"

namespace yy {

class IVolumeCUDA;

namespace volren {

class VolRenRaycastRAF : public VolRenRaycast
{
public:
    VolRenRaycastRAF();
    static std::unique_ptr<VolRenRaycastRAF> create();
    virtual ~VolRenRaycastRAF();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setVolume(const std::shared_ptr<IVolume>& volume);
    virtual void setColormap(const std::shared_ptr<IColormap>& colormap);
    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void newFBOs(int w, int h);
    virtual void raycast();

protected:
    static const int defaultLayers = 8;
    static const int maxLayers = 8;
    std::shared_ptr<GLuint> rafPBO, depPBO;
    cudaGraphicsResource *entryRes, *exitRes, *rafRes, *depRes;
    std::shared_ptr<IVolumeCUDA> volume;
    std::shared_ptr<cudaArray> tfArr;
    int texWidth, texHeight, layers;
    std::shared_ptr<IColormap> colormap;
    bool preintegrate;
    IColormap::Filter tfFilter;
    float stepsize;
    QSharedPointer<QOpenGLTexture> tfTex;

private:
    void updateColormapResources();
    void newOutPBO(std::shared_ptr<GLuint> *outPBO, cudaGraphicsResource **outRes, int w, int h, int l);
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCASTRAF_H
