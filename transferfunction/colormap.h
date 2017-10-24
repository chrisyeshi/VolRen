#ifndef COLORMAP_H
#define COLORMAP_H

#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QColor>
#include <tfintegrater.h>
#ifdef ENABLE_CUDA
    #include <cuda_runtime.h>
    #include <cuda_gl_interop.h>
#endif // ENABLE_CUDA

namespace yy {
namespace volren {

//
//
// Helper classes
//
//

class Rgba
{
public:
    Rgba() { for (auto& d : data) d = 0.f; }
    Rgba(float r, float g, float b, float a) { data[0] = r; data[1] = g; data[2] = b; data[3] = a; }
    static int nFloats() { return 4; }
    float r() const { return data[0]; }
    float& r() { return data[0]; }
    float g() const { return data[1]; }
    float& g() { return data[1]; }
    float b() const { return data[2]; }
    float& b() { return data[2]; }
    float a() const { return data[3]; }
    float& a() { return data[3]; }
    float operator[](int i) const { return data[i]; }
    float& operator[](int i) { return data[i]; }

    operator QColor() const
    {
        QColor ret;
        ret.setRgbF(data[0], data[1], data[2], data[3]);
        return ret;
    }

private:
    float data[4];
};

//
//
// Interfaces
//
//

class IColormap
{
public:
    enum Filter { Filter_Linear, Filter_Nearest };
    virtual const std::vector<Rgba>& buffer() const = 0;
    virtual float basesize() const = 0;
    virtual float stepsize() const = 0;
    virtual bool preintegrate() const = 0;
    virtual Filter filter() const = 0;

public:
    const float* bufPtr() const { return reinterpret_cast<const float*>(buffer().data()); }
    int nColors() const { return buffer().size(); }
    int nFloatsPerColor() const { return Rgba::nFloats(); }
    int nFloats() const { return nColors() * nFloatsPerColor(); }
    int nBytesPerColor() const { return nFloatsPerColor() * sizeof(float); }
    int nBytes() const { return nColors() * nBytesPerColor(); }
};

class IColormapGL : public virtual IColormap
{
public:
    virtual QSharedPointer<QOpenGLTexture> texFull() const = 0;
    virtual QSharedPointer<QOpenGLTexture> texBack() const = 0;
};

#ifdef ENABLE_CUDA
class IColormapCUDA : public virtual IColormap
{
public:
    virtual cudaGraphicsResource* cudaResFull() const = 0;
    virtual cudaGraphicsResource* cudaResBack() const = 0;
};
#endif // ENABLE_CUDA

//
//
// Implementations
//
//

class Colormap : public virtual IColormap
{
public:
    Colormap() : _buffer(3), _stepsize(0.01f), _preintegrate(false), _filter(Filter_Linear)
    {
        _buffer[0] = Rgba(0.2f, 0.2f, 1.0f, 0.1f);
        _buffer[1] = Rgba(0.8f, 1.0f, 0.1f, 0.1f);
        _buffer[2] = Rgba(1.0f, 0.0f, 0.1f, 0.1f);
    }
	virtual const std::vector<Rgba>& buffer() const { return _buffer; }
    virtual float basesize() const { return _basesize; }
	virtual float stepsize() const { return _stepsize; }
	virtual bool preintegrate() const { return _preintegrate; }
	virtual Filter filter() const { return _filter; }

private:
	std::vector<Rgba> _buffer;
    float _basesize;
	float _stepsize;
	bool _preintegrate;
	Filter _filter;
};

class ColormapGL : public virtual IColormapGL
{
public:
	ColormapGL() : _colormap(new Colormap()) {}
	ColormapGL(std::shared_ptr<IColormap> colormap) : _colormap(colormap) {}

public:
	virtual const std::vector<Rgba>& buffer() const { return _colormap->buffer(); }
    virtual float basesize() const { return _colormap->basesize(); }
	virtual float stepsize() const { return _colormap->stepsize(); }
	virtual bool preintegrate() const { return _tfInteg->isPreinteg(); }
	virtual Filter filter() const { return _colormap->filter(); }
	virtual QSharedPointer<QOpenGLTexture> texFull() const
	{
		integrate();
		return _tfInteg->getTexFull();
	}
	virtual QSharedPointer<QOpenGLTexture> texBack() const
	{
		integrate();
		return _tfInteg->getTexBack();
	}

private:
	std::shared_ptr<IColormap> _colormap;
	mutable std::shared_ptr<TFIntegrater> _tfInteg;

private:
	void integrate() const
	{
		if (_tfInteg)
			return;
		_tfInteg = std::make_shared<yy::volren::TFIntegrater>(true);
        _tfInteg->integrate(reinterpret_cast<const float*>(_colormap->buffer().data()), _colormap->buffer().size(), _colormap->basesize(), _colormap->stepsize());
		_tfInteg->getTexFull()->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
		_tfInteg->getTexFull()->setWrapMode(QOpenGLTexture::ClampToEdge);
		_tfInteg->getTexBack()->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
		_tfInteg->getTexBack()->setWrapMode(QOpenGLTexture::ClampToEdge);
	}
};

#ifdef ENABLE_CUDA
class ColormapGLCUDA : public virtual IColormapGL, public virtual IColormapCUDA
{
public:
    ColormapGLCUDA() : _colormap(new ColormapGL()), _cudaResFull(nullptr), _cudaResBack(nullptr) {}
    ColormapGLCUDA(std::shared_ptr<IColormap> colormap) : _colormap(nullptr), _cudaResFull(nullptr), _cudaResBack(nullptr)
	{
		std::shared_ptr<IColormapGL> ptr = std::dynamic_pointer_cast<IColormapGL>(colormap);
		if (!ptr)
			ptr = std::make_shared<ColormapGL>(colormap);
		_colormap = ptr;
	}
	virtual ~ColormapGLCUDA()
	{
	    if (_cudaResFull)
	    {
	        cudaGraphicsUnregisterResource(_cudaResFull);
	        _cudaResFull = nullptr;
	    }
	    if (_cudaResBack)
	    {
	        cudaGraphicsUnregisterResource(_cudaResBack);
	        _cudaResBack = nullptr;
	    }
	}

public:
	virtual const std::vector<Rgba>& buffer() const { return _colormap->buffer(); }
    virtual float basesize() const { return _colormap->basesize(); }
	virtual float stepsize() const { return _colormap->stepsize(); }
	virtual bool preintegrate() const { return _colormap->preintegrate(); }
	virtual Filter filter() const { return _colormap->filter(); }
	virtual QSharedPointer<QOpenGLTexture> texFull() const { return _colormap->texFull(); }
	virtual QSharedPointer<QOpenGLTexture> texBack() const { return _colormap->texBack(); }
	virtual cudaGraphicsResource* cudaResFull() const
	{
		if (!_cudaResFull)
			cudaGraphicsGLRegisterImage(&_cudaResFull, texFull()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
		return _cudaResFull;
	}
	virtual cudaGraphicsResource* cudaResBack() const
	{
		if (!_cudaResBack)
			cudaGraphicsGLRegisterImage(&_cudaResBack, texBack()->textureId(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
		return _cudaResBack;
	}

private:
	std::shared_ptr<IColormapGL> _colormap;
    mutable cudaGraphicsResource* _cudaResFull;
    mutable cudaGraphicsResource* _cudaResBack;
};
#endif // ENABLE_CUDA

} // namespace volren
} // namespace yy

#endif // COLORMAP_H
