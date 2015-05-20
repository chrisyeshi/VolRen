#include "volrenraycastcuda.h"
#include "volrenraycastcuda.cuda.h"
#include <cassert>
#include "imagepbo.h"
#include "volume.h"

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

VolRenRaycastCuda::VolRenRaycastCuda()
 : VolRenRaycast(Method_Raycast_CUDA)
 , outPBO(0)
 , entryRes(NULL), exitRes(NULL), outRes(NULL)
 , texWidth(defaultFBOSize), texHeight(defaultFBOSize)
 , volRes(0)
 , tfRes(0)
{

}

VolRenRaycastCuda::~VolRenRaycastCuda()
{
    if (entryRes) cc(cudaGraphicsUnregisterResource(entryRes));
    if (exitRes)  cc(cudaGraphicsUnregisterResource(exitRes));
    if (outRes)   cc(cudaGraphicsUnregisterResource(outRes));
    if (volRes)   cc(cudaGraphicsUnregisterResource(volRes));
    if (tfRes)    cc(cudaGraphicsUnregisterResource(tfRes));
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    if (0 != outPBO) f.glDeleteBuffers(1, &outPBO);
}

std::shared_ptr<ImageAbstract> VolRenRaycastCuda::output() const
{
    return std::make_shared<ImagePBO>(outPBO, texWidth, texHeight);
}

void VolRenRaycastCuda::newFBOs(int w, int h)
{
    this->texWidth = w;
    this->texHeight = h;
    newReadFBO(w, h, &entryFBO, &entryTex, &entryRen, &entryRes);
    newReadFBO(w, h, &exitFBO, &exitTex, &exitRen, &exitRes);
    newWriteFBO(w, h, &outPBO, &outRes);
}

void VolRenRaycastCuda::raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&)
{
    cc(cudaGraphicsMapResources(1, &entryRes, 0));
    cc(cudaGraphicsMapResources(1, &exitRes, 0));
    cc(cudaGraphicsMapResources(1, &outRes, 0));
    cc(cudaGraphicsMapResources(1, &volRes, 0));
    cc(cudaGraphicsMapResources(1, &tfRes, 0));

    cudaArray *entryArr, *exitArr, *volArr, *tfArr;
    float *outPtr;
    size_t nBytes;
    cc(cudaGraphicsSubResourceGetMappedArray(&entryArr, entryRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&exitArr, exitRes, 0, 0));
    cc(cudaGraphicsResourceGetMappedPointer((void**)&outPtr, &nBytes, outRes));
    cc(cudaGraphicsSubResourceGetMappedArray(&volArr, volRes, 0, 0));
    cc(cudaGraphicsSubResourceGetMappedArray(&tfArr, tfRes, 0, 0));

    static std::map<Filter, cudaTextureFilterMode> vr2cu
            = { { Filter_Linear, cudaFilterModeLinear }
              , { Filter_Nearest, cudaFilterModePoint } };
    assert(vr2cu.count(tfFilter) > 0);

    cudacast(vol()->w(), vol()->h(), vol()->d(), volArr,
             tfInteg->w(), tfInteg->h(), stepsize, vr2cu[tfFilter], tfArr,
             scalarMin, scalarMax,
             texWidth, texHeight, entryArr, exitArr, outPtr);

    cc(cudaGraphicsUnmapResources(1, &tfRes, 0));
    cc(cudaGraphicsUnmapResources(1, &volRes, 0));
    cc(cudaGraphicsUnmapResources(1, &entryRes, 0));
    cc(cudaGraphicsUnmapResources(1, &exitRes, 0));
    cc(cudaGraphicsUnmapResources(1, &outRes, 0));
}

void VolRenRaycastCuda::newReadFBO(int w, int h, std::shared_ptr<GLuint> *fbo, std::shared_ptr<GLuint> *tex, std::shared_ptr<GLuint> *ren, cudaGraphicsResource **res) const
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // texture
    tex->reset([](){
        GLuint* texPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenTextures(1, texPtr);
        return texPtr;
    }(), [](GLuint* texPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteTextures(1, texPtr);
    });
    f.glBindTexture(GL_TEXTURE_2D, **tex);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    f.glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT, NULL);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    // register cuda resource
    if (*res) cc(cudaGraphicsUnregisterResource(*res));
    cc(cudaGraphicsGLRegisterImage(res, **tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
    // renderbuffer
    ren->reset([](){
        GLuint* renPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenRenderbuffers(1, renPtr);
        return renPtr;
    }(), [](GLuint* renPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteRenderbuffers(1, renPtr);
    });
    f.glBindRenderbuffer(GL_RENDERBUFFER, **ren);
    f.glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, w, h);
    f.glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // fbo
    fbo->reset([](){
        GLuint* fboPtr = new GLuint();
        QOpenGLContext::currentContext()->functions()->glGenFramebuffers(1, fboPtr);
        return fboPtr;
    }(), [](GLuint* fboPtr){
        QOpenGLContext::currentContext()->functions()->glDeleteFramebuffers(1, fboPtr);
    });
    f.glBindFramebuffer(GL_FRAMEBUFFER, **fbo);
    f.glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, **tex, 0);
    f.glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, **ren);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // check framebuffer completeness
    GLenum status;
    status = f.glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (GL_FRAMEBUFFER_COMPLETE != status)
        std::cout << "framebuffer incomplete" << std::endl;
}

void VolRenRaycastCuda::newWriteFBO(int w, int h, GLuint *pbo, cudaGraphicsResource** res) const
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    // clean up
    if (0 != *pbo) f.glDeleteBuffers(1, pbo);
    // pbo
    f.glGenBuffers(1, pbo);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    f.glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * w * h * sizeof(float), NULL, GL_STREAM_COPY);
    f.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // register cuda resource
    if (*res) cc(cudaGraphicsUnregisterResource(*res));
    cc(cudaGraphicsGLRegisterBuffer(res, *pbo, cudaGraphicsMapFlagsWriteDiscard));
}

void VolRenRaycastCuda::volumeChanged()
{
    if (volRes) cc(cudaGraphicsUnregisterResource(volRes));
    cc(cudaGraphicsGLRegisterImage(&volRes, volTex->textureId(), GL_TEXTURE_3D, cudaGraphicsRegisterFlagsReadOnly));
}

void VolRenRaycastCuda::tfChanged(const mslib::TF &, bool, float, Filter)
{
    if (tfRes) cc(cudaGraphicsUnregisterResource(tfRes));
    cc(cudaGraphicsGLRegisterImage(&tfRes, tfTex->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly));
}
