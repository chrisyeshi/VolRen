// When linking as a library, add Q_INIT_RESOURCE(volren) to main().

#ifndef VOLREN_H
#define VOLREN_H

#include <QMatrix>
#include <QJsonObject>
#include <memory>
#include <TF.h>
#include <imageabstract.h>
#include "light.h"

// TODO: profile the performance

namespace yy {

class IVolume;

namespace volren {

enum Method { Method_Raycast_GL, Method_Raycast_CUDA, Method_Raycast_RAF, Method_Unknown };
enum Filter { Filter_Linear, Filter_Nearest };

class VolRen
{
public:
    static std::unique_ptr<VolRen> create(const Method& method);
    VolRen(const Method& method) : scalarMin(0.f), scalarMax(1.f), method(method) {}
    virtual ~VolRen() {}

    Method getMethod() const { return method; }
	virtual void initializeGL() = 0;
	virtual void resize(int w, int h) = 0;
    virtual void setVolume(const std::weak_ptr<IVolume>& volume) = 0;
    virtual void setTF(const mslib::TF& tf, bool preinteg, float stepsize, Filter filter) = 0;
    virtual void setScalarRange(float min, float max) { scalarMin = min; scalarMax = max; }
    virtual void setLights(const std::vector<Light>& lights) { this->lights = lights; }
    virtual void render(const QMatrix4x4& v, const QMatrix4x4& p) = 0;
    virtual std::shared_ptr<ImageAbstract> output() const = 0;

protected:
    float scalarMin, scalarMax;
    std::vector<Light> lights;

private:
    Method method;

    VolRen(); // Not implemented
};

} // namespace volren
} // namespace yy

#endif // VOLREN_H
