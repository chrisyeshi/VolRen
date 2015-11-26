//#ifndef VOLRENRAYCASTCUDA_H
//#define VOLRENRAYCASTCUDA_H

//#include <volrenraycast.h>
//#include <tfintegrated.h>
//#include <QtOpenGL>
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>

//namespace yy {
//namespace volren {

//class VolRenRaycastCuda : public TFIntegrated<VolRenRaycast>
//{
//public:
//    VolRenRaycastCuda();
//    virtual ~VolRenRaycastCuda();

//    virtual void initializeGL();
//    virtual void resize(int w, int h);
//    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, Filter filter);
//    virtual std::shared_ptr<ImageAbstract> output() const;

//protected:
//    virtual void raycast(const QMatrix4x4&, const QMatrix4x4& matView, const QMatrix4x4&);
//    virtual void volumeChanged();

//private:
//    void updateCUDAResources();
//    void updateCUDALights(const QMatrix4x4& matView);

//protected:
//    GLuint outPBO;
//    cudaGraphicsResource *entryRes, *exitRes, *outRes;
//    int texWidth, texHeight;
//    cudaGraphicsResource *volRes;
//    cudaGraphicsResource *tfFullRes, *tfBackRes;
//};

//} // namespace volren
//} // namespace yy

//#endif // VOLRENRAYCASTCUDA_H
