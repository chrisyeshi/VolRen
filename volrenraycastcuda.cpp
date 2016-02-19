#include "volrenraycastcuda.h"
#include "volrenraycastcuda.cuda.h"
#include <cassert>
#include <volumeglcuda.h>
#include <raycastfrustum.h>
#include "imagepbo.h"

//
//
// cuda error checking
//
//

#define cc(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//
//
// VolRenRaycastCuda
//
//

namespace yy {
namespace volren {

#define BASE VolRenRaycast

VolRenRaycastCuda::VolRenRaycastCuda()
 : BASE(Method_Raycast_CUDA)
 , outPBO(0)
 , entryRes(NULL), exitRes(NULL), outRes(NULL)
 , texWidth(_frustum->texWidth()), texHeight(_frustum->texHeight())
{

}

std::unique_ptr<VolRenRaycastCuda> VolRenRaycastCuda::create()
{
    return std::unique_ptr<VolRenRaycastCuda>(new VolRenRaycastCuda());
}

VolRenRaycastCuda::~VolRenRaycastCuda()
{
    if (entryRes)  cc(cudaGraphicsUnregisterResource(entryRes));
    if (exitRes)   cc(cudaGraphicsUnregisterResource(exitRes));
    if (outRes)    cc(cudaGraphicsUnregisterResource(outRes));
    // if (volRes)    cc(cudaGraphicsUnregisterResource(volRes));
    // if (tfFullRes) cc(cudaGraphicsUnregisterResource(tfFullRes));
    // if (tfBackRes) cc(cudaGraphicsUnregisterResource(tfBackRes));
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    if (0 != outPBO) f.glDeleteBuffers(1, &outPBO);
}

void VolRenRaycastCuda::initializeGL()
{
    BASE::initializeGL();
    updateCUDAResources();
}

void VolRenRaycastCuda::resize(int w, int h)
{
    BASE::resize(w, h);
    updateCUDAResources();
}

// TODO: SetVolumeCUDAed<BASE>
void VolRenRaycastCuda::setVolume(const std::shared_ptr<IVolume> &volume)
{
    // check supported pixel type
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume->pixelType()))
    {
        std::cout << "Unsupported pixel type..." << std::endl;
        return;
    }
    // TODO: convert directly from Volume to VolumeCUDA without VolumeGL
    // whether volume has the interface I need
    std::shared_ptr<IVolumeCUDA> ptr = std::dynamic_pointer_cast<IVolumeCUDA>(volume);
    if (!ptr)
        ptr.reset(new VolumeGLCUDA(volume));
    this->volume = ptr;
    // set bounding box dimension
    _frustum->setVolSize(
        this->volume->w() * this->volume->sx(),
        this->volume->h() * this->volume->sy(),
        this->volume->d() * this->volume->sz());
}

//void VolRenRaycastCuda::setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter)
//{
//    BASE::setTF(tf, preinteg, stepsize, filter);
//    if (tfFullRes) cc(cudaGraphicsUnregisterResource(tfFullRes));
//    cc(cudaGraphicsGLRegisterImage(&tfFullRes, tfInteg->getTexFull()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
//    if (tfBackRes) cc(cudaGraphicsUnregisterResource(tfBackRes));
//    cc(cudaGraphicsGLRegisterImage(&tfBackRes, tfInteg->getTexBack()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
//}

void VolRenRaycastCuda::setColormap(const std::shared_ptr<IColormap>& colormap)
{
    auto ptr = std::dynamic_pointer_cast<IColormapCUDA>(colormap);
    if (!ptr)
        ptr = std::make_shared<ColormapGLCUDA>(colormap);
    this->colormap = ptr;
}

std::shared_ptr<ImageAbstract> VolRenRaycastCuda::output() const
{
    return std::make_shared<ImagePBO>(outPBO, _frustum->texWidth(), _frustum->texHeight());
}

void VolRenRaycastCuda::raycast()
{
    // TODO: getCUDAMappedArray();
    cudaGraphicsResource_t volRes = this->volume->getCUDAResource();
    cudaGraphicsResource_t tfFullRes = this->colormap->cudaResFull();
    cudaGraphicsResource_t tfBackRes = this->colormap->cudaResBack();

    cc(cudaGraphicsMapResources(1, &entryRes, 0));
    cc(cudaGraphicsMapResources(1, &exitRes, 0));
    cc(cudaGraphicsMapResources(1, &outRes, 0));
    cc(cudaGraphicsMapResources(1, &volRes, 0));
    cc(cudaGraphicsMapResources(1, &tfFullRes, 0));
    cc(cudaGraphicsMapResources(1, &tfBackRes, 0));

    cudaArray *entryArr, *exitArr, *volArr, *tfFullArr, *tfBackArr;
    float *outPtr;
    size_t nBytes;
    cc(cudaGraphicsSubResourceGetMappedArray(&entryArr, entryRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&exitArr, exitRes, 0, 0));
    cc(cudaGraphicsResourceGetMappedPointer((void**)&outPtr, &nBytes, outRes));
    cc(cudaGraphicsSubResourceGetMappedArray(&volArr, volRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&tfFullArr, tfFullRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&tfBackArr, tfBackRes, 0, 0));

    static std::map<IColormap::Filter, cudaTextureFilterMode> vr2cu
            = { { IColormap::Filter_Linear, cudaFilterModeLinear }
              , { IColormap::Filter_Nearest, cudaFilterModePoint } };
    assert(vr2cu.count(colormap->filter()) > 0);

    updateCUDALights(_frustum->matView());

    cudacast(this->volume->w(), this->volume->h(), this->volume->d(), volArr,
             colormap->nColors(), colormap->nColors(), colormap->stepsize(), vr2cu[colormap->filter()], tfFullArr, tfBackArr,
             scalarMin, scalarMax,
             _frustum->texWidth(), _frustum->texHeight(), entryArr, exitArr, outPtr);

    cc(cudaGraphicsUnmapResources(1, &tfBackRes, 0));
    cc(cudaGraphicsUnmapResources(1, &tfFullRes, 0));
    cc(cudaGraphicsUnmapResources(1, &volRes, 0));
    cc(cudaGraphicsUnmapResources(1, &entryRes, 0));
    cc(cudaGraphicsUnmapResources(1, &exitRes, 0));
    cc(cudaGraphicsUnmapResources(1, &outRes, 0));
}

void VolRenRaycastCuda::updateCUDAResources()
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // update entry and exit resources
    if (entryRes) cc(cudaGraphicsUnregisterResource(entryRes));
    cc(cudaGraphicsGLRegisterImage(&entryRes, *_frustum->texEntry(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    if (exitRes)  cc(cudaGraphicsUnregisterResource(exitRes));
    cc(cudaGraphicsGLRegisterImage(&exitRes,  *_frustum->texExit(),  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    // update output PBO and CUDA resource
    // clean up
    if (0 != outPBO) f.glDeleteBuffers(1, &outPBO);
    // pbo
    f.glGenBuffers(1, &outPBO);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, outPBO);
    f.glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * _frustum->texWidth() * _frustum->texHeight() * sizeof(float), NULL, GL_STREAM_COPY);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // register cuda resource
    if (outRes) cc(cudaGraphicsUnregisterResource(outRes));
    cc(cudaGraphicsGLRegisterBuffer(&outRes, outPBO, cudaGraphicsMapFlagsWriteDiscard));
}

void VolRenRaycastCuda::updateCUDALights(const QMatrix4x4& matView)
{
    CudaLight cudaLights[10];
    for (unsigned int i = 0; i < lights.size(); ++i)
    {
        QVector3D lightDir(matView.inverted() * QVector4D(lights[i].direction, 0.f));
        lightDir.normalize();
        cudaLights[i].direction.x = lightDir.x();
        cudaLights[i].direction.y = lightDir.y();
        cudaLights[i].direction.z = lightDir.z();
        cudaLights[i].ambient.x = lights[i].color.x() * lights[i].ambient;
        cudaLights[i].ambient.y = lights[i].color.y() * lights[i].ambient;
        cudaLights[i].ambient.z = lights[i].color.z() * lights[i].ambient;
        cudaLights[i].diffuse.x = lights[i].color.x() * lights[i].diffuse;
        cudaLights[i].diffuse.y = lights[i].color.y() * lights[i].diffuse;
        cudaLights[i].diffuse.z = lights[i].color.z() * lights[i].diffuse;
        cudaLights[i].specular.x = lights[i].color.x() * lights[i].specular;
        cudaLights[i].specular.y = lights[i].color.y() * lights[i].specular;
        cudaLights[i].specular.z = lights[i].color.z() * lights[i].specular;
        cudaLights[i].shininess = lights[i].shininess;
    }
    cudaSetLights(lights.size(), cudaLights);
}

} // namespace volren
} // namespace yy
