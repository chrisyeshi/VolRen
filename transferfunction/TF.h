/*
 * This file is part of HPGVis which is released under MIT.
 * See file LICENSE for full license details.
 */

#ifndef TF_H
#define TF_H

#include <colormap.h>
#include <tfintegrater.h>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifndef nullptr
#define nullptr 0
#endif

namespace mslib {

struct TFColorControl
{
    TFColorControl() : value(0.0f) { setColor(1.0f, 1.0f, 1.0f); }
    TFColorControl(float _value, float r, float g, float b) : value(_value) { setColor(r, g, b); }
    bool operator < (const TFColorControl &other) const { return (value < other.value); }
    void setColor(float r, float g, float b) { color[0] = r; color[1] = g; color[2] = b; }
    float value;
    float color[3];
};

struct TFGaussianObject
{
    TFGaussianObject() : mean(0.5f), sigma(1.0f), heightFactor(1.0f), resolution(1024), alphaArray(nullptr) {}
    TFGaussianObject(float _mean, float _sigma, float _heightFactor, int _resolution) : mean(_mean), sigma(_sigma), heightFactor(_heightFactor), resolution(_resolution), alphaArray(nullptr) {}
    TFGaussianObject(const TFGaussianObject &other) { *this = other; }
    ~TFGaussianObject() { if (alphaArray != nullptr) delete [] alphaArray; }
    TFGaussianObject &operator = (const TFGaussianObject &other);
    float value(float x) const;
    float height() const { return value(mean); }
    void setHeight(float h);
    void update();
    float  mean;
    float  sigma;
    float  heightFactor;
    int    resolution;
    float *alphaArray;
};

inline TFGaussianObject &TFGaussianObject::operator = (const TFGaussianObject &other)
{
    mean = other.mean;
    sigma = other.sigma;
    heightFactor = other.heightFactor;
    resolution = other.resolution;
    alphaArray = nullptr;
    update();
    return *this;
}

inline float TFGaussianObject::value(float x) const
{
    float diff = x - mean;
    return heightFactor / (sigma * sqrt(2.0f * 3.14159265358979323846f)) * exp(-(diff * diff) / (2.0f * sigma * sigma));
}

inline void TFGaussianObject::setHeight(float h)
{
    heightFactor = h * sigma * sqrt(2.0f * 3.14159265358979323846f);
}

inline void TFGaussianObject::update()
{
    if (alphaArray == nullptr)
        alphaArray = new float[resolution];
    float invRes = 1.0f / (float)resolution;
    for (int i = 0; i < resolution; i++)
    {
        float val = value(((float)i + 0.5f) * invRes);
        alphaArray[i] = (val > 1.0f ? 1.0f : (val < 0.0f ? 0.0f : val));
    }
}

////////////////////////////////////////////////////////////////////////////////

//
// defines transfer function
//
// arraySize : internal array size of the hand-drawn alpha array
// resolution : sampling resolution of the output RGBA color map
//
class TF : public virtual yy::volren::IColormapGL, public virtual yy::volren::IColormapCUDA
{
public:
    TF(int resolution = 1024, int arraySize = 1024);
    TF(const TF &other);
    ~TF();

    TF &operator = (const TF &other);       // deep copy

    void setResolution(int resolution);
    void clear();
    void setAlpha(unsigned int index, float alpha) { if (index < _arraySize) _alphaArray[index] = alpha; }

    int resolution() const               { return _resolution; }
    float *colorMap()                    { return _colorMap; }
    const float *colorMap() const        { return _colorMap; }
    int arraySize() const                { return _arraySize; }
    float *alphaArray()                  { return _alphaArray; }
    const float *backgroundColor() const { return _backgroundColor; }
    void setBackgroundColor(float r, float g, float b);

    int                   colorControlCount() const     { return (int)_colorControls.size(); }
    TFColorControl       &colorControl(int index)       { return _colorControls[index]; }
    const TFColorControl &colorControl(int index) const { return _colorControls[index]; }
    void                  addColorControl(const TFColorControl &control);
    void                  removeColorControl(int index);
    TFColorControl       &insertColorControl(float value);
    void                  clearColorControls() { _colorControls.clear(); }
    
    int                     gaussianObjectCount() const     { return (int)_gaussianObjects.size(); }
    TFGaussianObject       &gaussianObject(int index)       { return _gaussianObjects[index]; }
    const TFGaussianObject &gaussianObject(int index) const { return _gaussianObjects[index]; }
    void                    addGaussianObject(float mean, float sigma, float heightFactor);
    void                    removeGaussianObject(int index);

    void updateColorMap();

    bool read(const char *fileName);
    bool write(const char *fileName) const;
    bool load(const char *fileName) { return read(fileName); }      // = read()
    bool save(const char *fileName) { return write(fileName); }     // = write()

    static TF fromRainbowMap(int resolution = 1024, int arraySize = 1024);

public:
    void setPreIntegrate(bool preinteg);
    void setStepsize(float stepsize) { _stepsize = stepsize; _isUpdatedGL = false; }
    void setFilter(yy::volren::IColormap::Filter filter) { _filter = filter; _isUpdatedGL = false; }
    virtual const std::vector<yy::volren::Rgba>& buffer() const;
    virtual float stepsize() const { return _stepsize; }
    virtual bool preintegrate() const;
    virtual yy::volren::IColormap::Filter filter() const { return _filter; }
    virtual QSharedPointer<QOpenGLTexture> texFull() const;
    virtual QSharedPointer<QOpenGLTexture> texBack() const;
    virtual cudaGraphicsResource* cudaResFull() const;
    virtual cudaGraphicsResource* cudaResBack() const;

private:
    int     _resolution;                            // sampling resolution of color map
    float * _colorMap;                              // rgba
    int     _arraySize;                             // size of alpha array, can be different from resolution
    float * _alphaArray;                            // alpha
    float   _backgroundColor[4];
    int     _blendMode;
    std::vector<TFColorControl> _colorControls;     // color control points
    std::vector<TFGaussianObject> _gaussianObjects;
    mutable bool _isUpdatedGL;
    float _stepsize;
    yy::volren::IColormap::Filter _filter;
    mutable std::shared_ptr<yy::volren::TFIntegrater> _tfInteg;
    mutable cudaGraphicsResource* _cudaResFull;
    mutable cudaGraphicsResource* _cudaResBack;

private:
    void tfIntegrate() const;
};

} // namespace mslib

#endif // TF_H
