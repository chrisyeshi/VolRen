#ifndef COLORMAP_H
#define COLORMAP_H

#include <QSharedPointer>
#include <QOpenGLTexture>
#include <tfintegrater.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace yy {
namespace volren {

class Rgba
{
public:
	Rgba() : r(0.f), g(0.f), b(0.f), a(0.f) {}
	Rgba(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
    float r, g, b, a;
    static int nFloats() { return 4; }
};

class IColormap
{
public:
    enum Filter { Filter_Linear, Filter_Nearest };
    virtual const std::vector<Rgba>& buffer() const = 0;
    int resolution() const { return buffer().size(); }
    virtual float stepsize() const = 0;
    virtual bool preintegrate() const = 0;
    virtual Filter filter() const = 0;
};

class IColormapGL : public virtual IColormap
{
public:
    virtual QSharedPointer<QOpenGLTexture> texFull() const = 0;
    virtual QSharedPointer<QOpenGLTexture> texBack() const = 0;
};

class IColormapCUDA : public virtual IColormap
{
public:
    virtual cudaGraphicsResource* cudaResFull() const = 0;
    virtual cudaGraphicsResource* cudaResBack() const = 0;
};

class Colormap : public virtual IColormap
{
public:
    Colormap() : _buffer(3), _stepsize(0.01f), _preintegrate(false), _filter(Filter_Linear)
    {
    	_buffer[0] = Rgba(0.2f, 0.2f, 1.0f, 0.0f);
    	_buffer[1] = Rgba(0.8f, 1.0f, 0.1f, 0.5f);
    	_buffer[2] = Rgba(1.0f, 0.0f, 0.1f, 1.0f);
    }
	virtual const std::vector<Rgba>& buffer() const { return _buffer; }
	virtual float stepsize() const { return _stepsize; }
	virtual bool preintegrate() const { return _preintegrate; }
	virtual Filter filter() const { return _filter; }

private:
	std::vector<Rgba> _buffer;
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
		_tfInteg->integrate(reinterpret_cast<const float*>(_colormap->buffer().data()), _colormap->buffer().size(), _colormap->stepsize());
		_tfInteg->getTexFull()->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
		_tfInteg->getTexFull()->setWrapMode(QOpenGLTexture::ClampToEdge);
		_tfInteg->getTexBack()->setMinMagFilters(QOpenGLTexture::Linear, QOpenGLTexture::Linear);
		_tfInteg->getTexBack()->setWrapMode(QOpenGLTexture::ClampToEdge);
	}
};

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

} // namespace volren
} // namespace yy

#endif // COLORMAP_H
