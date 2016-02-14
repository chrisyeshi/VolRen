// When linking as a library, add Q_INIT_RESOURCE(volren) to main().

#ifndef VOLREN_H
#define VOLREN_H

#include <QMatrix>
#include <memory>
#include <map>
#include <functional>
#include <TF.h>
#include <imageabstract.h>
#include "light.h"

// TODO: profile the performance

namespace yy {

class IVolume;

namespace volren {

enum Method { Method_Raycast_GL, Method_Raycast_CUDA, Method_Raycast_RAF, Method_Unknown };

class VolRen
{
public:
    static std::map<Method, std::string> method2string;

public:
    VolRen(const Method& method) : scalarMin(0.f), scalarMax(1.f), method(method) {}
    virtual ~VolRen() {}

public:
    Method getMethod() const { return method; }
	virtual void initializeGL() = 0;
	virtual void resize(int w, int h) = 0;
    virtual void setVolume(const std::shared_ptr<IVolume>& volume) = 0;
    virtual void setColormap(const std::shared_ptr<IColormap>& colormap) = 0;
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

class VolRenFactory
{
public:
    typedef std::function<std::unique_ptr<VolRen>()> CreateFunc;
    static std::vector<Method> methods();
    static std::unique_ptr<VolRen> create(const Method& method);

private:
    static std::map<Method, CreateFunc> creators;
};

class VolRenNull : public VolRen
{
public:
    VolRenNull() : VolRen(Method_Unknown) {}
    static std::unique_ptr<VolRenNull> create() { return std::unique_ptr<VolRenNull>(new VolRenNull()); }
    virtual ~VolRenNull() {}

    virtual void initializeGL() {}
    virtual void resize(int w, int h) {}
    virtual void setVolume(const std::shared_ptr<IVolume>& volume) {}
    virtual void setColormap(const std::shared_ptr<IColormap>& colormap) {}
    virtual void render(const QMatrix4x4& v, const QMatrix4x4& p) {}
    virtual std::shared_ptr<ImageAbstract> output() const { return std::make_shared<ImageNull>(); }
};

} // namespace volren
} // namespace yy

#endif // VOLREN_H
