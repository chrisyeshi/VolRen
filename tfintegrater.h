#ifndef TFINTEGRATER_H
#define TFINTEGRATER_H

#include <memory>
#include <QOpenGLTexture>
#include <QSharedPointer>

class TFIntegrater
{
public:
    static std::unique_ptr<TFIntegrater> create(bool preintegrate);
    TFIntegrater();
    virtual ~TFIntegrater();

    virtual QSharedPointer<QOpenGLTexture> newTexture(int size) = 0;
    virtual const std::unique_ptr<float[]>& integrate(QSharedPointer<QOpenGLTexture> tex, const float* colormap, float stepsize) = 0;
    virtual const std::unique_ptr<float[]>& output() const { return tf; }
    virtual int w() const = 0;
    virtual int h() const = 0;

protected:
    std::unique_ptr<float[]> tf;

private:
};

#endif // TFINTEGRATER_H
