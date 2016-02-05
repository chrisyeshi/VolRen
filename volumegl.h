#ifndef VOLUMEGL_H
#define VOLUMEGL_H

#include "volume.h"
#include "volren.h"
#include <memory>
#include <QSharedPointer>
#include <QOpenGLTexture>

namespace yy {
namespace volren {

class VolumeGL : public IVolume
{
public:
    VolumeGL(const std::shared_ptr<IVolume>& volume) : volume(volume), filter(Filter_Linear) {}
    virtual ~VolumeGL();

    friend std::ostream& operator<<(std::ostream& os, const VolumeGL& volume);
    virtual int w() const { return volume->w(); }
    virtual int h() const { return volume->h(); }
    virtual int d() const { return volume->d(); }
    virtual float sx() const { return volume->sx(); }
    virtual float sy() const { return volume->sy(); }
    virtual float sz() const { return volume->sz(); }
    virtual ScalarType scalarType() const { return volume->scalarType(); }
    virtual unsigned int nBytesPerScalar() const { return volume->nBytesPerScalar(); }
    virtual unsigned int nScalarsPerVoxel() const { return volume->nScalarsPerVoxel(); }
    virtual const Stats& getStats() const { return volume->getStats(); }
    virtual const std::unique_ptr<unsigned char []>& getData() const { return volume->getData(); }
    virtual void normalized() { volume->normalized(); }
    virtual void makeTexture();
    virtual QSharedPointer<QOpenGLTexture> getTexture() const { return texture; }
    virtual void setFilter(Filter filter);
    virtual Filter getFilter() const { return filter; }

private:
    std::shared_ptr<IVolume> volume;
    QSharedPointer<QOpenGLTexture> texture;
    Filter filter;

    static std::map<Filter, QOpenGLTexture::Filter> filter2qgl;
};

} // namespace volren
} // namespace yy

#endif // VOLUMEGL_H
