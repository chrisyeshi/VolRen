/*
 * This file is part of HPGVis which is released under MIT.
 * See file LICENSE for full license details.
 */

#include <algorithm>

// I/O
#include <iostream>
#include <fstream>
#include <string>

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

TF::TF(int resolution, int arraySize)
    : _resolution(resolution),
      _colorMap(nullptr),
      _arraySize(arraySize),
      _alphaArray(nullptr),
      _blendMode(0)
{
    _backgroundColor[0] = 0.0f;
    _backgroundColor[1] = 0.0f;
    _backgroundColor[2] = 0.0f;
    _backgroundColor[3] = 1.0f;
    
    if (_resolution > 0)
    {
        _resolution = resolution;
        _colorMap = new float[resolution * 4];      // rgba
        _arraySize = arraySize;
        _alphaArray = new float[arraySize]();       // alpha
    
        _colorControls.push_back(TFColorControl(0.0f, 13.f/255.f, 132.f/255.f, 211.f/255.f));
        _colorControls.push_back(TFColorControl(0.5f, 244.f/255.f, 208.f/255.f, 27.f/255.f));
        _colorControls.push_back(TFColorControl(1.0f, 194.f/255.f, 75.f/255.f, 64.f/255.f));

        // initial draw array
        for (int i = 0; i < arraySize; i++)
            _alphaArray[i] = 0.5f;

        updateColorMap();
    }
}

TF::TF(const TF &other)
    : _resolution(0),
      _colorMap(nullptr),
      _arraySize(0),
      _alphaArray(nullptr),
      _blendMode(0)
{
    *this = other;
}

TF::~TF()
{
    clear();
}

TF &TF::operator = (const TF &other)
{
    if (_colorMap != nullptr) delete [] _colorMap;
    std::cout << _alphaArray << std::endl;
    if (_alphaArray != nullptr) delete [] _alphaArray;
    _resolution = other._resolution;
    _colorMap = new float[_resolution * 4];
    for (int i = 0; i < _resolution * 4; i++)
        _colorMap[i] = other._colorMap[i];
    _arraySize = other._arraySize;
    _alphaArray = new float[_arraySize];
    for (int i = 0; i < _arraySize; i++)
        _alphaArray[i] = other._alphaArray[i];
    _backgroundColor[0] = other._backgroundColor[0];
    _backgroundColor[1] = other._backgroundColor[1];
    _backgroundColor[2] = other._backgroundColor[2];
    _backgroundColor[3] = other._backgroundColor[3];
    _colorControls = other._colorControls;
    _gaussianObjects = other._gaussianObjects;
    return *this;
}

void TF::clear()
{
    if (_colorMap != nullptr) delete [] _colorMap;
    if (_alphaArray != nullptr) delete [] _alphaArray;
    _colorMap = nullptr;
    _alphaArray = nullptr;
    _resolution = 0;
    _arraySize = 0;
    _backgroundColor[0] = 0.0f;
    _backgroundColor[1] = 0.0f;
    _backgroundColor[2] = 0.0f;
    _backgroundColor[3] = 1.0f;
    _colorControls.clear();
    _gaussianObjects.clear();
}

void TF::setBackgroundColor(float r, float g, float b)
{
    _backgroundColor[0] = r;
    _backgroundColor[1] = g;
    _backgroundColor[2] = b;
    _backgroundColor[3] = 1.0f;
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
    _gaussianObjects.push_back(TFGaussianObject(mean, sigma, heightFactor, _resolution));
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
    
    for (int i = 0; i < _resolution; i++)
    {
        float value = (float)indexToValue(i, _resolution);
        
        // find the first color control that is larger than the value
        while (firstLarger < controlCount && value > colorControls[firstLarger].value)
        {
            firstLarger++;
        }

        if (firstLarger <= 0)                       // less than the smallest control point
        {
            _colorMap[i * 4]     = colorControls[firstLarger].color[0];
            _colorMap[i * 4 + 1] = colorControls[firstLarger].color[1];
            _colorMap[i * 4 + 2] = colorControls[firstLarger].color[2];
        }
        else if (firstLarger >= controlCount)       // greater than the largest control point
        {
            _colorMap[i * 4]     = colorControls[firstLarger - 1].color[0];
            _colorMap[i * 4 + 1] = colorControls[firstLarger - 1].color[1];
            _colorMap[i * 4 + 2] = colorControls[firstLarger - 1].color[2];
        }
        else                                        // between two control points
        {
            TFColorControl &left = colorControls[firstLarger - 1];
            TFColorControl &right = colorControls[firstLarger];
            float w = std::abs(value - left.value) / std::abs(right.value - left.value);
            _colorMap[i * 4]     = lerp(left.color[0], right.color[0], w);
            _colorMap[i * 4 + 1] = lerp(left.color[1], right.color[1], w);
            _colorMap[i * 4 + 2] = lerp(left.color[2], right.color[2], w);
        }
        
        if (_resolution == _arraySize)
            _colorMap[i * 4 + 3] = _alphaArray[i];
        else if (_resolution * 2 == _arraySize)
        {
            //_colorMap[i * 4 + 3] = (_alphaArray[i * 2] + _alphaArray[i * 2 + 1]) * 0.5f;
            _colorMap[i * 4 + 3] = _alphaArray[i * 2];
        }
        else
        {
            // sample from alpha array
            int f = (int)floor(value * _arraySize - 0.5f);
            int c = f + 1;
            f = fmax(f, 0);
            c = fmin(c, _arraySize - 1);
            float w = fmax(value * _arraySize - ((int)f + 0.5f), 0.0f);
            _colorMap[i * 4 + 3] = lerp(_alphaArray[f], _alphaArray[c], w);
        }

        for (int j = 0; j < (int)_gaussianObjects.size(); j++)
            if (_colorMap[i * 4 + 3] < _gaussianObjects[j].alphaArray[i])       // maximum
                _colorMap[i * 4 + 3] = _gaussianObjects[j].alphaArray[i];
    }
}

// static
TF TF::fromRainbowMap(int resolution, int arraySize)
{
    TF ret(0);      // empty TF

    ret._resolution = resolution;
    ret._colorMap = new float[resolution * 4];      // rgba
    ret._arraySize = arraySize;
    ret._alphaArray = new float[arraySize]();
    
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
