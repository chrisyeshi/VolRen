#ifndef TFINTEGRATED_H
#define TFINTEGRATED_H

#include <tfintegrater.h>
#include <QOpenGLTexture>
#include <cassert>
#include <volren.h>
#include <TF.h>

namespace yy {
namespace volren {

template <typename BASE>
class TFIntegrated : public BASE
{
public:
    TFIntegrated(Method method) : BASE(method), tfInteg(new TFIntegrater()) {}
    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, Filter filter)
    {
        BASE::setTF(tf, preinteg, stepsize, filter);
        tfInteg->convertTo(preinteg);
        tfInteg->integrate(tf.colorMap(), tf.resolution(), stepsize);
        static std::map<Filter, QOpenGLTexture::Filter> vr2qt
                = { { Filter_Linear, QOpenGLTexture::Linear }
                  , { Filter_Nearest, QOpenGLTexture::Nearest } };
        assert(vr2qt.count(filter) > 0);
        tfInteg->getTexture()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
        tfInteg->getTexture()->setWrapMode(QOpenGLTexture::ClampToEdge);
    }

protected:
    std::unique_ptr<TFIntegrater> tfInteg;

private:
    TFIntegrated(); // Not implemented!!!
};

} // namespace volren
} // namespace yy

#endif // TFINTEGRATED_H
