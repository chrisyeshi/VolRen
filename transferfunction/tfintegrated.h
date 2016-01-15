#ifndef TFINTEGRATED_H
#define TFINTEGRATED_H

#include <tfintegrater.h>
#include <QOpenGLTexture>
#include <cassert>
#include <string>
#include <json/json.h>
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
      , tf(1024, 1024), tfFilter(Filter_Linear), stepsize(0.01f) {}

    virtual void initializeGL()
    {
        BASE::initializeGL();
        setTF(tf, tfInteg->isPreinteg(), stepsize, tfFilter);
    }

    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, Filter filter)
    {
        this->tf = tf;
        this->tfFilter = filter;
        this->stepsize = stepsize;
        tfInteg->convertTo(preinteg);
        tfInteg->integrate(tf.colorMap(), tf.resolution(), stepsize);
        static std::map<Filter, QOpenGLTexture::Filter> vr2qt
                = { { Filter_Linear, QOpenGLTexture::Linear }
                  , { Filter_Nearest, QOpenGLTexture::Nearest } };
        assert(vr2qt.count(filter) > 0);
        tfInteg->getTexFull()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
        tfInteg->getTexFull()->setWrapMode(QOpenGLTexture::ClampToEdge);
        tfInteg->getTexBack()->setMinMagFilters(vr2qt[filter], vr2qt[filter]);
        tfInteg->getTexBack()->setWrapMode(QOpenGLTexture::ClampToEdge);
    }

    virtual void setParaSheet(const Json::Value& json)
    {
        BASE::setParaSheet(json);
        if (!json.isMember("TransferFunction"))
            return;
        const Json::Value& tfJson = json["TransferFunction"];
        const int nRgba = 4;
        mslib::TF ntf = this->tf;
        bool nPreinteg = this->tfInteg->isPreinteg();
        Filter nTfFilter = this->tfFilter;
        float nStepsize = this->stepsize;
        if (tfJson.isMember("Colormap"))
        {
            Json::Value cm = tfJson["Colormap"];
            if (cm.isArray())
            {
                ntf = mslib::TF(tfJson["Colormap"].size(), tfJson["Colormap"].size());
                for (int iColor = 0; iColor < tfJson["Colormap"].size(); ++iColor)
                for (int iValue = 0; iValue < nRgba; ++iValue)
                    ntf.colorMap()[nRgba * iColor + iValue] = tfJson["Colormap"][iColor][iValue].asFloat();
            } else if (cm.isString())
            {
                const int nRgba = 4;
                std::string cms = tfJson["Colormap"].asString();
                int resolution = cms.size() / sizeof(float) / nRgba;
                ntf = mslib::TF(resolution, resolution);
                memcpy(reinterpret_cast<char*>(ntf.colorMap()), cms.data(), cms.size());
            }
        }
        if (tfJson.isMember("Preinteg"))
            nPreinteg = tfJson["Preinteg"].asBool();
        if (tfJson.isMember("Filter"))
        {
            static std::map<std::string, Filter> string2filter
                    = { { "Filter_Nearest", Filter_Nearest },
                        { "Filter_Linear",  Filter_Linear } };
            nTfFilter = string2filter[tfJson["Filter"].asString()];
        }
        if (tfJson.isMember("Stepsize"))
            nStepsize = tfJson["Stepsize"].asFloat();
        this->setTF(ntf, nPreinteg, nStepsize, nTfFilter);
    }

    virtual Json::Value getParaSheet() const
    {
        Json::Value ret = BASE::getParaSheet();
        const int nRgba = 4;
        const char* cmp = reinterpret_cast<const char*>(tf.colorMap());
        std::string cms(cmp, cmp + tf.resolution() * nRgba * sizeof(float));
        ret["TransferFunction"]["Colormap"] = cms;
//        for (int iColor = 0; iColor < tf.resolution(); ++iColor)
//        for (int iValue = 0; iValue < nRgba; ++iValue)
//            ret["TransferFunction"]["Colormap"][iColor][iValue] = tf.colorMap()[nRgba * iColor + iValue];
        static std::map<Filter, std::string> filter2string
                = { { Filter_Nearest, "Filter_Nearest" },
                    { Filter_Linear,  "Filter_Linear" } };
        ret["TransferFunction"]["Preinteg"] = tfInteg->isPreinteg();
        ret["TransferFunction"]["Filter"] = filter2string[tfFilter];
        ret["TransferFunction"]["Stepsize"] = stepsize;
        return ret;
    }

protected:
    std::unique_ptr<TFIntegrater> tfInteg;
    mslib::TF tf;
    Filter tfFilter;
    float stepsize;

private:
    TFIntegrated(); // Not implemented!!!
};

} // namespace volren
} // namespace yy

#endif // TFINTEGRATED_H
