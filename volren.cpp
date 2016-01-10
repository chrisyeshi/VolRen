#include "volren.h"
#include <cassert>
#include <json/json.h>
#include "volrenraycastgl.h"
#ifdef ENABLE_CUDA
    #include "volrenraycastcuda.h"
    #include "volrenraycastraf.h"
#endif // ENABLE_CUDA

namespace yy {
namespace volren {

static std::map<Method, std::string> method2string
        = { { Method_Raycast_CUDA, "Method_Raycast_CUDA" },
            { Method_Raycast_GL,   "Method_Raycast_GL" },
            { Method_Raycast_RAF,  "Method_Raycast_RAF" },
            { Method_Unknown,      "Method_Unknown" } };

void VolRen::setParaSheet(const Json::Value &json)
{
    if (json.isMember("Method"))
        assert(json["Method"].asString() == method2string[this->method]);
    if (json.isMember("ScalarRange"))
        this->setScalarRange(json["ScalarRange"][0].asFloat(), json["ScalarRange"][1].asFloat());
    if (json.isMember("Lights"))
    {
        lights.resize(json["Lights"].size());
        for (unsigned int iLight = 0; iLight < lights.size(); ++iLight)
            lights[iLight].fromJson(json["Lights"][iLight]);
    }
}

Json::Value VolRen::getParaSheet() const
{
    Json::Value ret;
    ret["Method"] = method2string[this->method];
    ret["ScalarRange"][0] = this->scalarMin;
    ret["ScalarRange"][1] = this->scalarMax;
    Json::Value jsonLights;
    for (auto light : lights)
        jsonLights.append(light.toJson());
    ret["Lights"] = jsonLights;
    return ret;
}

// TODO: automatic registration. C++ doesn't support static initialization in a static library (lame)
std::map<Method, VolRenFactory::CreateFunc> VolRenFactory::creators
  = { { Method_Raycast_GL,   VolRenRaycastGL::create }
#ifdef ENABLE_CUDA
    , { Method_Raycast_CUDA, VolRenRaycastCuda::create }
    , { Method_Raycast_RAF,  VolRenRaycastRAF::create }
#endif // ENABLE_CUDA
    };

std::vector<Method> VolRenFactory::methods()
{
    std::vector<Method> ret;
    for (auto entry : creators)
        ret.push_back(entry.first);
    return ret;
}

std::unique_ptr<VolRen> VolRenFactory::create(const Method &method)
{
    if (0 == creators.count(method))
        return std::unique_ptr<VolRen>(new VolRenNull());
    return creators[method]();
}

} // namespace volren
} // namespace yy
