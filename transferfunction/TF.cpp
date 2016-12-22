/*
 * This file is part of HPGVis which is released under MIT.
 * See file LICENSE for full license details.
 */

#include <algorithm>

// I/O
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include "TF.h"

namespace mslib {

namespace {

template <class T>
inline T fmin(T a, T b)
{
    return !(b < a) ? a : b;
}

template <class T>
inline T fmax(T a, T b)
{
    return (a < b) ? b : a;
}

inline float lerp(float a, float b, float w)
{
    return (a + w * (b - a));
}

inline double indexToValue(int index, int resolution)
{
    return (((double)index + 0.5) / (double)resolution);
}

} // namespace

TF::TF(int resolution)
  : _colorMap(resolution)
  , _alphaArray(resolution)
  , _blendMode(0)
  , _isUpdatedGL(false)
  , _basesize(0.01f)
  , _stepsize(0.01f)
  , _preintegrate(true)
  , _filter(Filter_Linear)
{
    _colorControls.push_back(TFColorControl(0.0f, 13.f/255.f, 132.f/255.f, 211.f/255.f));
    _colorControls.push_back(TFColorControl(0.5f, 244.f/255.f, 208.f/255.f, 27.f/255.f));
    _colorControls.push_back(TFColorControl(1.0f, 194.f/255.f, 75.f/255.f, 64.f/255.f));
    for (auto& alpha : _alphaArray)
        alpha = 0.5f;
    updateColorMap();
}

TF::TF(const TF &other)
  : _colorMap(0)
  , _alphaArray(0)
  , _blendMode(0)
  , _isUpdatedGL(false)
  , _stepsize(0.01f)
  , _preintegrate(true)
  , _filter(Filter_Linear)
{
    *this = other;
}

TF::~TF()
{
    clear();
}

TF &TF::operator = (const TF &other)
{
    _colorMap = other._colorMap;
    _alphaArray = other._alphaArray;
    _backgroundColor = other._backgroundColor;
    _colorControls = other._colorControls;
    _gaussianObjects = other._gaussianObjects;
    _stepsize = other._stepsize;
    _filter = other._filter;
    return *this;
}

void TF::clear()
{
    _colorMap.clear();
    _alphaArray.clear();
    _backgroundColor = yy::volren::Rgba();
    _colorControls.clear();
    _gaussianObjects.clear();
    _isUpdatedGL = false;
    _tfInteg.reset();
}

void TF::setResolution(int resolution)
{
    // TODO: resample when changing resolution instead of reinitialize
    assert(resolution > 0);
    _colorMap.resize(resolution);
    // resample alpha array
    std::vector<float> alphaArray(resolution);
    for (unsigned int iNew = 0; iNew < alphaArray.size(); ++iNew)
    {
        // nearest sampling
        int iOld = float(iNew) / float(alphaArray.size() - 1) * float(_alphaArray.size() - 1) + 0.5f;
        alphaArray[iNew] = _alphaArray[iOld];
    }
    _alphaArray = std::move(alphaArray);
    updateColorMap();
}

void TF::setBackgroundColor(float r, float g, float b)
{
    _backgroundColor.r() = r;
    _backgroundColor.g() = g;
    _backgroundColor.b() = b;
    _backgroundColor.a() = 1.0f;
}

void TF::addColorControl(const TFColorControl &control)
{
    _colorControls.push_back(control);
    updateColorMap();
}

void TF::removeColorControl(int index)
{
    if (index < 0 || index >= (int)_colorControls.size())
        return;
    for (int i = index + 1; i < (int)_colorControls.size(); i++)
        _colorControls[i - 1] = _colorControls[i];
    _colorControls.pop_back();
    updateColorMap();
}

TFColorControl &TF::insertColorControl(float value)
{
    TFColorControl control;
    control.value = value;

    std::vector<TFColorControl> colorControls = _colorControls;
    std::sort(colorControls.begin(), colorControls.end());
    int controlCount = (int)colorControls.size();
    int firstLarger = 0;
    while (firstLarger < controlCount && value > colorControls[firstLarger].value)
        firstLarger++;

    if (firstLarger <= 0)                       // less than the smallest control point
    {
        control.color[0] = colorControls[firstLarger].color[0];
        control.color[1] = colorControls[firstLarger].color[1];
        control.color[2] = colorControls[firstLarger].color[2];
    }
    else if (firstLarger >= controlCount)       // greater than the largest control point
    {
        control.color[0] = colorControls[firstLarger - 1].color[0];
        control.color[1] = colorControls[firstLarger - 1].color[1];
        control.color[2] = colorControls[firstLarger - 1].color[2];
    }
    else
    {
        TFColorControl &left = colorControls[firstLarger - 1];
        TFColorControl &right = colorControls[firstLarger];
        float w = std::abs(value - left.value) / std::abs(right.value - left.value);
        control.color[0] = lerp(left.color[0], right.color[0], w);
        control.color[1] = lerp(left.color[1], right.color[1], w);
        control.color[2] = lerp(left.color[2], right.color[2], w);
    }

    _colorControls.push_back(control);
    updateColorMap();

    return _colorControls.back();
}

void TF::addGaussianObject(float mean, float sigma, float heightFactor)
{
    _gaussianObjects.push_back(TFGaussianObject(mean, sigma, heightFactor, nColors()));
    updateColorMap();
}

void TF::removeGaussianObject(int index)
{
    if (index < 0 || index >= (int)_gaussianObjects.size())
        return;                                                 // exception: index out of bounds
    for (int i = index + 1; i < (int)_gaussianObjects.size(); i++)
        _gaussianObjects[i - 1] = _gaussianObjects[i];
    _gaussianObjects.pop_back();
    updateColorMap();
}

void TF::updateColorMap()
{
    if (_colorControls.size() < 1)
        return;                     // no valid color map

    std::vector<TFColorControl> colorControls = _colorControls;
    std::sort(colorControls.begin(), colorControls.end());
    int controlCount = (int)colorControls.size();
    int firstLarger = 0;
    int resolution = _colorMap.size();

    for (int i = 0; i < resolution; i++)
    {
        float value = (float)indexToValue(i, resolution);

        // find the first color control that is larger than the value
        while (firstLarger < controlCount && value > colorControls[firstLarger].value)
        {
            firstLarger++;
        }

        if (firstLarger <= 0)                       // less than the smallest control point
        {
            _colorMap[i].r() = colorControls[firstLarger].color[0];
            _colorMap[i].g() = colorControls[firstLarger].color[1];
            _colorMap[i].b() = colorControls[firstLarger].color[2];
        }
        else if (firstLarger >= controlCount)       // greater than the largest control point
        {
            _colorMap[i].r() = colorControls[firstLarger - 1].color[0];
            _colorMap[i].g() = colorControls[firstLarger - 1].color[1];
            _colorMap[i].b() = colorControls[firstLarger - 1].color[2];
        }
        else                                        // between two control points
        {
            TFColorControl &left = colorControls[firstLarger - 1];
            TFColorControl &right = colorControls[firstLarger];
            float w = std::abs(value - left.value) / std::abs(right.value - left.value);
            _colorMap[i].r() = lerp(left.color[0], right.color[0], w);
            _colorMap[i].g() = lerp(left.color[1], right.color[1], w);
            _colorMap[i].b() = lerp(left.color[2], right.color[2], w);
        }
        _colorMap[i].a() = _alphaArray[i];

        for (int j = 0; j < (int)_gaussianObjects.size(); j++)
            if (_colorMap[i].a() < _gaussianObjects[j].alphaArray[i])       // maximum
                _colorMap[i].a() = _gaussianObjects[j].alphaArray[i];
    }

    for (unsigned int iIso = 0; iIso < _isoValues.size(); ++iIso) {
        auto iso = _isoValues[iIso];
        int midIndex = iso * (resolution - 1);
        float alpha = 0.99f;
        const int thickness = 3;
        int startIndex = std::max(0, midIndex - thickness / 2);
        int endIndex = std::min(resolution - 1, midIndex + (thickness - thickness / 2));
        for (int index = startIndex; index <= endIndex; ++index)
            _colorMap[index].a() = std::max(alpha, _colorMap[index].a());
    }

    _isUpdatedGL = false;
}

// static
TF TF::fromRainbowMap(int resolution)
{
    TF ret(resolution);      // empty TF

    ret._colorControls.push_back(TFColorControl(0.0f / 6.0f, 0.0f,      0.364706f, 1.0f));
    ret._colorControls.push_back(TFColorControl(1.0f / 6.0f, 0.0f,      1.0f,      0.976471f));
    ret._colorControls.push_back(TFColorControl(2.0f / 6.0f, 0.0f,      1.0f,      0.105882f));
    ret._colorControls.push_back(TFColorControl(3.0f / 6.0f, 0.968627f, 1.0f,      0.0f));
    ret._colorControls.push_back(TFColorControl(4.0f / 6.0f, 1.0f,      0.490196f, 0.0f));
    ret._colorControls.push_back(TFColorControl(5.0f / 6.0f, 1.0f,      0.0f,      0.0f));
    ret._colorControls.push_back(TFColorControl(6.0f / 6.0f, 0.662745f, 0.0f,      1.0f));

    // gaussian
    float hf = 0.5f * 0.1f * sqrt(2.0f * 3.14159265358979323846f);
    ret._gaussianObjects.push_back(TFGaussianObject(0.5f, 0.1f, hf, resolution));
    ret._gaussianObjects[0].update();

    ret.updateColorMap();
    return ret;
}

void TF::setPreIntegrate(bool preinteg)
{
    if (_preintegrate == preinteg)
        return;
    _preintegrate = preinteg;
    _isUpdatedGL = false;\
}

const std::vector<yy::volren::Rgba>& TF::buffer() const
{
    return _colorMap;
}

bool TF::preintegrate() const
{
    return _preintegrate;
}

QSharedPointer<QOpenGLTexture> TF::texFull() const
{
    if (!_tfInteg || !_isUpdatedGL)
        tfIntegrate();
    return _tfInteg->getTexFull();
}

QSharedPointer<QOpenGLTexture> TF::texBack() const
{
    if (!_tfInteg || !_isUpdatedGL)
        tfIntegrate();
    return _tfInteg->getTexBack();
}

#ifdef ENABLE_CUDA
cudaGraphicsResource* TF::cudaResFull() const
{
    if (!_cudaResFull || !_isUpdatedGL)
    {
        _cudaResFull.reset([this]() {
            cudaGraphicsResource* ptr = nullptr;
            cudaGraphicsGLRegisterImage(&ptr, texFull()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
            return ptr;
        }(), [](cudaGraphicsResource* ptr) {
            cudaGraphicsUnregisterResource(ptr);
            ptr = nullptr;
        });
    }
    return _cudaResFull.get();
}

cudaGraphicsResource* TF::cudaResBack() const
{
    if (!_cudaResBack || !_isUpdatedGL)
    {
        _cudaResBack.reset([this]() {
            cudaGraphicsResource* ptr = nullptr;
            cudaGraphicsGLRegisterImage(&ptr, texBack()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
            return ptr;
        }(), [](cudaGraphicsResource* ptr) {
            cudaGraphicsUnregisterResource(ptr);
            ptr = nullptr;
        });
    }
    return _cudaResBack.get();
}
#endif // ENABLE_CUDA

void TF::tfIntegrate() const
{
    if (!_tfInteg)
        _tfInteg = std::make_shared<yy::volren::TFIntegrater>();
    _tfInteg->convertTo(_preintegrate);
    _tfInteg->integrate(reinterpret_cast<const float*>(_colorMap.data()), _colorMap.size(), _basesize, _stepsize);
    static std::map<Filter, QOpenGLTexture::Filter> vr2qt
                = { { Filter_Linear, QOpenGLTexture::Linear }
                  , { Filter_Nearest, QOpenGLTexture::Nearest } };
    assert(vr2qt.count(_filter) > 0);
    _tfInteg->getTexFull()->setMinMagFilters(vr2qt[_filter], vr2qt[_filter]);
    _tfInteg->getTexFull()->setWrapMode(QOpenGLTexture::ClampToEdge);
    _tfInteg->getTexBack()->setMinMagFilters(vr2qt[_filter], vr2qt[_filter]);
    _tfInteg->getTexBack()->setWrapMode(QOpenGLTexture::ClampToEdge);
    _isUpdatedGL = true;
}

namespace {

std::string getSuffix(const std::string &filePath)
{
    size_t dotPos = filePath.find_last_of(".");
    return (dotPos == std::string::npos) ? "" : filePath.substr(dotPos + 1);
}

}

bool TF::read(const char *fileName)
{
    std::string fname(fileName);
    std::string suffix = getSuffix(fname);

    return false;
}

bool TF::write(const char *fileName) const
{
    std::string fname(fileName);
    std::string suffix = getSuffix(fname);

    return false;
}

} // namespace mslib
