#ifndef RAYCASTCUBE_H
#define RAYCASTCUBE_H

#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include "painter.h"

namespace yy {
namespace volren {

class RaycastCube
{
public:
    RaycastCube();
    ~RaycastCube();

    void initializeGL();
    void setSize(int w, int h, int d);
    const QMatrix4x4 &matrix();
    void render(const QMatrix4x4& vp, GLenum face = GL_BACK);

protected:

private:
    yy::Painter painter;
    int width, height, depth;
};

} // namespace volren
} // namespace yy

#endif // RAYCASTCUBE_H
