#ifndef IMAGEABSTRACT_H
#define IMAGEABSTRACT_H

namespace yy {
namespace volren {

class ImageAbstract
{
public:
    enum Type { TYPE_FBO, TYPE_TEXTURE, TYPE_PBO, TYPE_PBO_RAF, TYPE_NULL };

	ImageAbstract(Type type) : _type(type) {}
    virtual ~ImageAbstract() {}

	Type type() const { return _type; }
    virtual void initialize() = 0;
    virtual void draw() = 0;

private:
	Type _type;

	ImageAbstract(); // Not implemented!!!
};

class ImageNull : public ImageAbstract
{
public:
	ImageNull() : ImageAbstract(TYPE_NULL) {}
	virtual void initialize() {}
	virtual void draw() {}
};

} // namespace volren
} // namespace yy

#endif // IMAGEABSTRACT_H
