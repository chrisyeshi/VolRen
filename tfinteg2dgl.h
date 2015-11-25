#ifndef TFINTEG2DGL_H
#define TFINTEG2DGL_H

#include "tfintegrater.h"
#include "painter.h"
#include <memory>
#include <vector>
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>

namespace yy {
namespace volren {

class TFInteg2DGL : public ITFIntegrater
{
public:
	TFInteg2DGL();
	virtual ~TFInteg2DGL();

public:
	virtual void integrate(const float* colormap, int resolution, float stepsize);
	virtual QSharedPointer<QOpenGLTexture> getTexFull() const { return texFull; }
	virtual QSharedPointer<QOpenGLTexture> getTexBack() const { return texBack; }

private:
    QSharedPointer<QOpenGLTexture> tex1d, texFull, texBack;
    std::shared_ptr<GLuint> fbo;
    Painter painter;

	void newResources(int resolution);
};

} // namespace volren
} // namespace yy

#endif // TFINTEG2DGL_H
