#ifndef IMAGEWRITERRAF_H
#define IMAGEWRITERRAF_H

#include "imagewriter.h"
#include "imageraf.h"

class ImageWriterRAF : public ImageWriter
{
public:
    ImageWriterRAF(std::shared_ptr<ImageRAF> image) : image(image) {}
    virtual ~ImageWriterRAF() {}

    virtual void write(const std::string &name) const;

private:
    std::shared_ptr<ImageRAF> image;

private:
    ImageWriterRAF(); // Not implemented yet!!!
};

#endif // IMAGEWRITERRAF_H
