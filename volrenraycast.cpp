#include "volrenraycast.h"
#include <map>
#include <cassert>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"
#include "volume.h"

namespace yy {
namespace volren {

VolRenRaycast::VolRenRaycast(const Method &method)
 : VolRen(method)
 , tfFilter(Filter_Linear)
 , preintegrate(false)
 , tfInteg(TFIntegrater::create(false))
 , stepsize(0.01f)
{

}

VolRenRaycast::~VolRenRaycast()
{
}

void VolRenRaycast::initializeGL()
{
    frustum.initializeGL();
//    cube.initializeGL();
//    newFBOs(defaultFBOSize, defaultFBOSize);
    this->setTF(mslib::TF(1024, 1024), preintegrate, stepsize, tfFilter);
}

void VolRenRaycast::resize(int w, int h)
{
    frustum.setResolution(w, h);
//    newFBOs(w, h);
}

void VolRenRaycast::setVolume(const std::weak_ptr<Volume> &volume)
{
    // if same volume
    if (this->volume.lock().get() == volume.lock().get())
        return;
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume.lock()->pixelType()))
    {
        std::cout << "Unsupported pixel type..." << std::endl;
        return;
    }
    this->volume = volume;
    // set cube dimension
    frustum.setVolumeDimension(vol()->w() * vol()->sx(), vol()->h() * vol()->sy(), vol()->d() * vol()->sz());
//    cube.setSize(vol()->w() * vol()->sx(), vol()->h() * vol()->sy(), vol()->d() * vol()->sz());
    // setup 3d texture
    volTex = QSharedPointer<QOpenGLTexture>(new QOpenGLTexture(QOpenGLTexture::Target3D));
    volTex->setFormat(QOpenGLTexture::R32F);
    volTex->setSize(vol()->w(), vol()->h(), vol()->d());
    volTex->allocateStorage();
    volTex->setData(QOpenGLTexture::Red, dt2pt[vol()->pixelType()], vol()->getData().get());
    volTex->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
    volTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    // volume changed
    volumeChanged();
}

void VolRenRaycast::setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter)
{
    if (this->preintegrate != preinteg)
    {
        this->preintegrate = preinteg;
        tfInteg = std::move(TFIntegrater::create(this->preintegrate));
        tfTex = tfInteg->newTexture(tf.resolution());

    } else if (tfTex.isNull() || tfTex->width() != tf.resolution())
    {
        tfTex = tfInteg->newTexture(tf.resolution());
    }
    tfFilter = filter;
    tfInteg->integrate(tfTex, tf.colorMap(), stepsize);
    static std::map<Filter, QOpenGLTexture::Filter> vr2qt
            = { { Filter_Linear, QOpenGLTexture::Linear }
              , { Filter_Nearest, QOpenGLTexture::Nearest } };
    assert(vr2qt.count(filter) > 0);
    tfTex->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
    tfTex->setWrapMode(QOpenGLTexture::ClampToEdge);
    this->stepsize = stepsize;
    tfChanged(tf, preinteg, stepsize, filter);
}

void VolRenRaycast::render(const QMatrix4x4& v, const QMatrix4x4 &p)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    frustum.entryTexture(v, p);
    frustum.exitTexture(v, p);
    // raycast
    raycast(frustum.modelMatrix(), v, p);
}

} // namespace volren
} // namespace yy

