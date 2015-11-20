#include "volrenraycastcuda.h"
#include "volrenraycastcuda.cuda.h"
#include <cassert>
#include "imagepbo.h"
#include "volumegl.h"

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

VolRenRaycastCuda::VolRenRaycastCuda()
 : TFIntegrated<VolRenRaycast>(Method_Raycast_CUDA)
 , outPBO(0)
 , entryRes(NULL), exitRes(NULL), outRes(NULL)
 , texWidth(defaultFBOSize), texHeight(defaultFBOSize)
 , volRes(0)
 , tfFullRes(0), tfBackRes(0)
{

}

VolRenRaycastCuda::~VolRenRaycastCuda()
{
    if (entryRes)  cc(cudaGraphicsUnregisterResource(entryRes));
    if (exitRes)   cc(cudaGraphicsUnregisterResource(exitRes));
    if (outRes)    cc(cudaGraphicsUnregisterResource(outRes));
    if (volRes)    cc(cudaGraphicsUnregisterResource(volRes));
    if (tfFullRes) cc(cudaGraphicsUnregisterResource(tfFullRes));
    if (tfBackRes) cc(cudaGraphicsUnregisterResource(tfBackRes));
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    if (0 != outPBO) f.glDeleteBuffers(1, &outPBO);
}

void VolRenRaycastCuda::initializeGL()
{
    VolRenRaycast::initializeGL();
    updateCUDAResources();
}

void VolRenRaycastCuda::resize(int w, int h)
{
    VolRenRaycast::resize(w, h);
    updateCUDAResources();
}

void VolRenRaycastCuda::setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter)
{
    TFIntegrated<VolRenRaycast>::setTF(tf, preinteg, stepsize, filter);
    if (tfFullRes) cc(cudaGraphicsUnregisterResource(tfFullRes));
    cc(cudaGraphicsGLRegisterImage(&tfFullRes, tfInteg->getTexFull()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    if (tfBackRes) cc(cudaGraphicsUnregisterResource(tfBackRes));
    cc(cudaGraphicsGLRegisterImage(&tfBackRes, tfInteg->getTexBack()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}

std::shared_ptr<ImageAbstract> VolRenRaycastCuda::output() const
{
    return std::make_shared<ImagePBO>(outPBO, frustum.getTextureWidth(), frustum.getTextureHeight());
}

void VolRenRaycastCuda::raycast(const QMatrix4x4&, const QMatrix4x4& matView, const QMatrix4x4&)
{
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

    static std::map<Filter, cudaTextureFilterMode> vr2cu
            = { { Filter_Linear, cudaFilterModeLinear }
              , { Filter_Nearest, cudaFilterModePoint } };
    assert(vr2cu.count(tfFilter) > 0);

    updateCUDALights(matView);

    cudacast(vol()->w(), vol()->h(), vol()->d(), volArr,
             tfInteg->getTexFull()->width(), tfInteg->getTexFull()->height(), stepsize, vr2cu[tfFilter], tfFullArr, tfBackArr,
             scalarMin, scalarMax,
             frustum.getTextureWidth(), frustum.getTextureHeight(), entryArr, exitArr, outPtr);

    cc(cudaGraphicsUnmapResources(1, &tfBackRes, 0));
    cc(cudaGraphicsUnmapResources(1, &tfFullRes, 0));
    cc(cudaGraphicsUnmapResources(1, &volRes, 0));
    cc(cudaGraphicsUnmapResources(1, &entryRes, 0));
    cc(cudaGraphicsUnmapResources(1, &exitRes, 0));
    cc(cudaGraphicsUnmapResources(1, &outRes, 0));
}

void VolRenRaycastCuda::volumeChanged()
{
    if (volRes) cc(cudaGraphicsUnregisterResource(volRes));
    cc(cudaGraphicsGLRegisterImage(&volRes, volume->getTexture()->textureId(), GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
}

void VolRenRaycastCuda::updateCUDAResources()
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // update entry and exit resources
    if (entryRes) cc(cudaGraphicsUnregisterResource(entryRes));
    cc(cudaGraphicsGLRegisterImage(&entryRes, *frustum.entryTexture(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    if (exitRes)  cc(cudaGraphicsUnregisterResource(exitRes));
    cc(cudaGraphicsGLRegisterImage(&exitRes,  *frustum.exitTexture(),  GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    // update output PBO and CUDA resource
    // clean up
    if (0 != outPBO) f.glDeleteBuffers(1, &outPBO);
    // pbo
    f.glGenBuffers(1, &outPBO);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, outPBO);
    f.glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * frustum.getTextureWidth() * frustum.getTextureHeight() * sizeof(float), NULL, GL_STREAM_COPY);
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
