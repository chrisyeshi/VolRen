#include "volrenraycast.h"
#include <map>
#include <cassert>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"
#include "volumegl.h"

namespace yy {
namespace volren {

VolRenRaycast::VolRenRaycast(const Method &method)
 : VolRen(method)
 , tfFilter(Filter_Linear)
// , preintegrate(false)
 , tfInteg(new TFIntegrater())
 , stepsize(0.01f)
{

}

VolRenRaycast::~VolRenRaycast()
{
}

void VolRenRaycast::initializeGL()
{
    frustum.initializeGL();
    this->setTF(mslib::TF(1024, 1024), true, stepsize, tfFilter);
}

void VolRenRaycast::resize(int w, int h)
{
    frustum.setResolution(w, h);
//    newFBOs(w, h);
}

void VolRenRaycast::setVolume(const std::weak_ptr<IVolume> &volume)
{
    // if same volume
    if (reinterpret_cast<void*>(this->volume.get()) == reinterpret_cast<void*>(volume.lock().get()))
        return;
    // check suppoted pixel type
    static std::map<Volume::DataType, QOpenGLTexture::PixelType> dt2pt
            = {{Volume::DT_Char, QOpenGLTexture::Int8},
               {Volume::DT_Unsigned_Char, QOpenGLTexture::UInt8},
               {Volume::DT_Float, QOpenGLTexture::Float32}};
    if (0 == dt2pt.count(volume.lock()->pixelType()))
    {
        std::cout << "Unsupported pixel type..." << std::endl;
        return;
    }
    // whether it's a shared volume
    std::shared_ptr<VolumeGL> shared = std::dynamic_pointer_cast<VolumeGL>(volume.lock());
    if (shared)
        this->volume = shared;
    else
        this->volume.reset(new VolumeGL(volume.lock()));
    // setup 3d texture
    if (!this->volume->getTexture())
        this->volume->makeTexture();
    // set cube dimension
    frustum.setVolumeDimension(vol()->w() * vol()->sx(), vol()->h() * vol()->sy(), vol()->d() * vol()->sz());
    // volume changed
    volumeChanged();
}

void VolRenRaycast::setTF(const mslib::TF &tf, bool preinteg, float stepsize, Filter filter)
{
    tfInteg->convertTo(preinteg);
    tfInteg->integrate(tf.colorMap(), tf.resolution(), stepsize);
    tfFilter = filter;
    static std::map<Filter, QOpenGLTexture::Filter> vr2qt
            = { { Filter_Linear, QOpenGLTexture::Linear }
              , { Filter_Nearest, QOpenGLTexture::Nearest } };
    assert(vr2qt.count(filter) > 0);
    tfInteg->getTexture()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
    tfInteg->getTexture()->setWrapMode(QOpenGLTexture::ClampToEdge);
    this->stepsize = stepsize;
    tfChanged(tf, preinteg, stepsize, filter);
}

void VolRenRaycast::render(const QMatrix4x4& v, const QMatrix4x4 &p)
{
    frustum.entryTexture(v, p);
    frustum.exitTexture(v, p);
    // raycast
    raycast(frustum.modelMatrix(), v, p);
}

} // namespace volren
} // namespace yy

