#include "volrenraycastraf.h"
#include "volrenraycastraf.cuda.h"
#include <cassert>
#include <QOpenGLFunctions_3_3_Core>
#include "imageraf.h"
#include "volumeglcuda.h"

#undef near
#undef far

namespace yy {
namespace volren {

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
// VolRenRaycastRAF
//
//

VolRenRaycastRAF::VolRenRaycastRAF()
 : VolRenRaycast(Method_Raycast_RAF)
 , entryRes(NULL), exitRes(NULL)
 , rafRes(NULL), depRes(NULL)
 , texWidth(_frustum.texWidth()), texHeight(_frustum.texHeight())
 , layers(defaultLayers)
 , preintegrate(true)
 , tfFilter(IColormap::Filter_Linear)
 , stepsize(0.01f)
{

}

std::unique_ptr<VolRenRaycastRAF> VolRenRaycastRAF::create()
{
    return std::unique_ptr<VolRenRaycastRAF>(new VolRenRaycastRAF());
}

VolRenRaycastRAF::~VolRenRaycastRAF()
{
    if (entryRes) cc(cudaGraphicsUnregisterResource(entryRes));
    if (exitRes)  cc(cudaGraphicsUnregisterResource(exitRes));
    if (rafRes)   cc(cudaGraphicsUnregisterResource(rafRes));
    if (depRes)   cc(cudaGraphicsUnregisterResource(depRes));
}

void VolRenRaycastRAF::initializeGL()
{
    VolRenRaycast::initializeGL();
    newFBOs(_frustum.texWidth(), _frustum.texHeight());
}

void VolRenRaycastRAF::resize(int w, int h)
{
    VolRenRaycast::resize(w, h);
    newFBOs(w, h);
}

// TODO: SetVolumeCUDAed<BASE>
void VolRenRaycastRAF::setVolume(const std::shared_ptr<IVolume>& volume)
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
    // whether volume has the interface I need
    std::shared_ptr<IVolumeCUDA> ptr = std::dynamic_pointer_cast<IVolumeCUDA>(volume);
    if (!ptr)
        ptr.reset(new VolumeGLCUDA(volume));
    this->volume = ptr;
    // set bounding box dimension
    _frustum.setVolSize(
        this->volume->w() * this->volume->sx(),
        this->volume->h() * this->volume->sy(),
        this->volume->d() * this->volume->sz());
}

void VolRenRaycastRAF::setColormap(const std::shared_ptr<IColormap>& colormap)
{
    this->colormap = colormap;
}

std::shared_ptr<ImageAbstract> VolRenRaycastRAF::output() const
{
    return std::make_shared<ImageRAF>(rafPBO, depPBO, texWidth, texHeight, layers, tfTex);
}

void VolRenRaycastRAF::newFBOs(int w, int h)
{
    texWidth = w;
    texHeight = h;
//    VolRenRaycast::newFBOs(texWidth, texHeight);
    cc(cudaGraphicsGLRegisterImage(&entryRes, *_frustum.texEntry(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    cc(cudaGraphicsGLRegisterImage(&exitRes, *_frustum.texExit(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    newOutPBO(&rafPBO, &rafRes, texWidth, texHeight, layers);
    newOutPBO(&depPBO, &depRes, texWidth, texHeight, layers);
}

void VolRenRaycastRAF::raycast(const QMatrix4x4& m, const QMatrix4x4& v, const QMatrix4x4& p)
{
    updateColormapResources();
    // TODO: getCUDAMappedArray();
    cudaGraphicsResource_t volRes = this->volume->getCUDAResource();

    cc(cudaGraphicsMapResources(1, &entryRes, 0));
    cc(cudaGraphicsMapResources(1, &exitRes, 0));
    cc(cudaGraphicsMapResources(1, &rafRes, 0));
    cc(cudaGraphicsMapResources(1, &depRes, 0));
    cc(cudaGraphicsMapResources(1, &volRes, 0));

    cudaArray *entryArr, *exitArr, *volArr;
    float *rafPtr, *depPtr, *mPtr, *mvPtr;
    size_t nBytes;
    cc(cudaGraphicsSubResourceGetMappedArray(&entryArr, entryRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&exitArr, exitRes, 0, 0));
    cc(cudaGraphicsResourceGetMappedPointer((void**)&rafPtr, &nBytes, rafRes));
    cc(cudaGraphicsResourceGetMappedPointer((void**)&depPtr, &nBytes, depRes));
    cc(cudaGraphicsSubResourceGetMappedArray(&volArr, volRes, 0, 0));
    // copy matrices
    cc(cudaMalloc(&mPtr, 16 * sizeof(float)));
    cc(cudaMemcpy(mPtr, m.constData(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    cc(cudaMalloc(&mvPtr, 16 * sizeof(float)));
    cc(cudaMemcpy(mvPtr, (v * m).constData(), 16 * sizeof(float), cudaMemcpyHostToDevice));
    // near far from projection matrix
    float a = p.column(2)[2];
    float b = p.column(3)[2];
    float near = b / (a - 1);
    float far = b / (a + 1);
    // transfer function filter mode
    static std::map<IColormap::Filter, cudaTextureFilterMode> vr2cu
            = { { IColormap::Filter_Linear, cudaFilterModeLinear }
              , { IColormap::Filter_Nearest, cudaFilterModePoint } };
    assert(vr2cu.count(tfFilter) > 0);
    // default binDivs
    std::vector<float> binDivs(layers + 1);
    binDivs[0] = 0.f;
    binDivs[layers] = 1.f;
    for (int i = 0; i < layers; ++i)
        binDivs[i] = float(i) * 1.f / float(layers);
    // cast
    rafcast(this->volume->w(), this->volume->h(), this->volume->d(), volArr,
            tfTex->width(), stepsize, vr2cu[tfFilter], preintegrate, tfArr.get(),
            scalarMin, scalarMax,
            texWidth, texHeight, entryArr, exitArr,
            layers, binDivs.data(), rafPtr,
            mPtr, mvPtr, near, far, depPtr);
    // clean matrices
    cc(cudaFree(mPtr));
    cc(cudaFree(mvPtr));

    cc(cudaGraphicsUnmapResources(1, &volRes, 0));
    cc(cudaGraphicsUnmapResources(1, &entryRes, 0));
    cc(cudaGraphicsUnmapResources(1, &exitRes, 0));
    cc(cudaGraphicsUnmapResources(1, &rafRes, 0));
    cc(cudaGraphicsUnmapResources(1, &depRes, 0));
}

void VolRenRaycastRAF::updateColormapResources()
{
    this->stepsize = colormap->stepsize();
    this->preintegrate = colormap->preintegrate();
    this->tfFilter = colormap->filter();
    if (tfTex.isNull() || tfTex->width() != colormap->nColors() || tfTex->height() != 1)
    {
        tfTex.reset(new QOpenGLTexture(QOpenGLTexture::Target1D));
        tfTex->setFormat(QOpenGLTexture::RGBA32F);
        tfTex->setSize(colormap->nColors());
        tfTex->allocateStorage();
        tfTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    }
    static std::map<IColormap::Filter, QOpenGLTexture::Filter> vr2qt
        = { { IColormap::Filter_Linear, QOpenGLTexture::Linear }
          , { IColormap::Filter_Nearest, QOpenGLTexture::Nearest } };
          assert(vr2qt.count(tfFilter) > 0);
    tfTex->setMinMagFilters(vr2qt[tfFilter], vr2qt[tfFilter]);
    tfTex->setData(QOpenGLTexture::RGBA, QOpenGLTexture::Float32, colormap->bufPtr());
    // unable to register image with 1D texture
    // creating cudaArray manually
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    tfArr.reset([&](){
        cudaArray* arr;
        cc(cudaMallocArray(&arr, &channelDesc, colormap->nColors()));
        cc(cudaMemcpyToArray(arr, 0, 0, colormap->bufPtr(), colormap->nBytes(), cudaMemcpyHostToDevice));
        return arr;
    }(), [](cudaArray* arr){
        cc(cudaFreeArray(arr));
    });
    // define number of layers for RAF
    int newLayers = std::min(colormap->nColors(), int(maxLayers));
    if (layers != newLayers)
    {
        layers = 8;
        newOutPBO(&rafPBO, &rafRes, texWidth, texHeight, layers);
        newOutPBO(&depPBO, &depRes, texWidth, texHeight, layers);
    }
}

void VolRenRaycastRAF::newOutPBO(std::shared_ptr<GLuint>* outPBO, cudaGraphicsResource** outRes, int w, int h, int l)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // using std::shared_ptr to manage the PBO memory
    outPBO->reset([](){
        GLuint* ptrPBO = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenBuffers(1, ptrPBO);
        return ptrPBO;
    }(), [](GLuint* ptrPBO){
        QOpenGLContext::currentContext()->functions()->glDeleteBuffers(1, ptrPBO);
    });
    // setup gl PBO
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, **outPBO);
    f.glBufferData(GL_PIXEL_UNPACK_BUFFER, w * h * l * sizeof(float), NULL, GL_STREAM_COPY);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // register CUDA resource
    if (*outRes) cc(cudaGraphicsUnregisterResource(*outRes));
    cc(cudaGraphicsGLRegisterBuffer(outRes, **outPBO, cudaGraphicsMapFlagsWriteDiscard));
}

} // namespace volren
} // namespace yy
