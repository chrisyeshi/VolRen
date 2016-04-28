#ifndef TFINTEGRATER_H
#define TFINTEGRATER_H

#include <memory>
#include <QOpenGLTexture>
#include <QSharedPointer>

namespace yy {
namespace volren {

class ITFIntegrater
{
public:
    virtual ~ITFIntegrater() {}
    virtual void integrate(const float* colormap, int resolution, float basesize, float stepsize) = 0;
    virtual QSharedPointer<QOpenGLTexture> getTexFull() const = 0;
    virtual QSharedPointer<QOpenGLTexture> getTexBack() const = 0;
};

class TFIntegrater : public ITFIntegrater
{
public:
    TFIntegrater();
    TFIntegrater(bool preinteg);
    virtual ~TFIntegrater() {}

    virtual void integrate(const float* colormap, int resolution, float basesize, float stepsize) { return integ->integrate(colormap, resolution, basesize, stepsize); }
    virtual QSharedPointer<QOpenGLTexture> getTexFull() const { return integ->getTexFull(); }
    virtual QSharedPointer<QOpenGLTexture> getTexBack() const { return integ->getTexBack(); }

    bool isPreinteg() const { return preinteg; }
    void convertTo(bool preinteg);

private:
    std::shared_ptr<ITFIntegrater> integ;
    bool preinteg;
};

} // namespace volren
} // namespace yy

#endif // TFINTEGRATER_H
