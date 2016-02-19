#ifndef VOLRENRAYCASTCUDA_H
#define VOLRENRAYCASTCUDA_H

#include <volrenraycast.h>
#include <tfintegrated.h>
#include <volumeglcuda.h>
#include <QtOpenGL>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace yy {
namespace volren {

class VolRenRaycastCuda : public VolRenRaycast
{
public:
    VolRenRaycastCuda();
    static std::unique_ptr<VolRenRaycastCuda> create();
    virtual ~VolRenRaycastCuda();

    virtual void initializeGL();
    virtual void resize(int w, int h);
    virtual void setVolume(const std::shared_ptr<IVolume> &volume);
    virtual void setColormap(const std::shared_ptr<IColormap>& colormap);
    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void raycast();

private:
    void updateCUDAResources();
    void updateCUDALights(const QMatrix4x4& matView);

protected:
    std::shared_ptr<IVolumeCUDA> volume;
    std::shared_ptr<IColormapCUDA> colormap;
    GLuint outPBO;
    cudaGraphicsResource *entryRes, *exitRes, *outRes;
    int texWidth, texHeight;
};

} // namespace volren
} // namespace yy

#endif // VOLRENRAYCASTCUDA_H
