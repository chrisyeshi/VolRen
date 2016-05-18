#ifndef TFINTEGRATED_H
#define TFINTEGRATED_H

#include <tfintegrater.h>
#include <QOpenGLTexture>
#include <cassert>
#include <string>
#include <volren.h>
#include <TF.h>

namespace yy {
namespace volren {

template <typename BASE>
class TFIntegrated : public BASE
{
public:
    TFIntegrated(Method method)
      : BASE(method), tfInteg(new TFIntegrater())
      , tf(1024), tfFilter(IColormap::Filter_Linear) {}

    virtual void initializeGL()
    {
        BASE::initializeGL();
        setTF(tf, tfInteg->isPreinteg(), tfFilter);
    }

    virtual void setTF(const mslib::TF& tf, bool preinteg, IColormap::Filter filter)
    {
        this->tf = tf;
        this->tfFilter = filter;
        tfInteg->convertTo(preinteg);
        tfInteg->integrate(tf.bufPtr(), tf.nColors(), tf.basesize(), tf.stepsize());
        static std::map<IColormap::Filter, QOpenGLTexture::Filter> vr2qt
                = { { IColormap::Filter_Linear, QOpenGLTexture::Linear }
                  , { IColormap::Filter_Nearest, QOpenGLTexture::Nearest } };
        assert(vr2qt.count(filter) > 0);
        tfInteg->getTexFull()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
        tfInteg->getTexFull()->setWrapMode(QOpenGLTexture::ClampToEdge);
        tfInteg->getTexBack()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
        tfInteg->getTexBack()->setWrapMode(QOpenGLTexture::ClampToEdge);
    }

protected:
    std::unique_ptr<TFIntegrater> tfInteg;
    mslib::TF tf;
    IColormap::Filter tfFilter;

private:
    TFIntegrated(); // Not implemented!!!
};

} // namespace volren
} // namespace yy

#endif // TFINTEGRATED_H
