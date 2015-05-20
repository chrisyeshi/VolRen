#ifndef VOLRENRAYCASTCUDA_H
#define VOLRENRAYCASTCUDA_H

#include "volrenraycast.h"
#include <QtOpenGL>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class VolRenRaycastCuda : public VolRenRaycast
{
public:
    VolRenRaycastCuda();
    virtual ~VolRenRaycastCuda();

    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void newFBOs(int w, int h);
    virtual void raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&);
    virtual void newReadFBO(int w, int h, std::shared_ptr<GLuint> *fbo, std::shared_ptr<GLuint> *tex, std::shared_ptr<GLuint> *ren, cudaGraphicsResource** res) const;
    virtual void newWriteFBO(int w, int h, GLuint *pbo, cudaGraphicsResource** res) const;
    virtual void volumeChanged();
    virtual void tfChanged(const mslib::TF&,bool,float,Filter);

protected:
    GLuint outPBO;
    cudaGraphicsResource *entryRes, *exitRes, *outRes;
    int texWidth, texHeight;
    cudaGraphicsResource *volRes;
    cudaGraphicsResource *tfRes;
};

#endif // VOLRENRAYCASTCUDA_H
