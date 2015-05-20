#ifndef IMAGEABSTRACT_H
#define IMAGEABSTRACT_H

class ImageAbstract
{
public:
    enum Type { TYPE_FBO, TYPE_TEXTURE, TYPE_PBO, TYPE_PBO_RAF };

	ImageAbstract(Type type) : _type(type) {}
    virtual ~ImageAbstract() {}

	Type type() const { return _type; }
    virtual void initialize() = 0;
    virtual void draw() = 0;

private:
	Type _type;

	ImageAbstract(); // Not implemented!!!
};

#endif // IMAGEABSTRACT_H
