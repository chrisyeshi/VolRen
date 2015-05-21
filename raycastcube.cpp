#include "raycastcube.h"
#include <cassert>
#include "shape.h"

RaycastCube::RaycastCube()
 : width(1), height(1), depth(1)
{

}

RaycastCube::~RaycastCube()
{

}

void RaycastCube::initializeGL()
{
    painter.initializeGL(yy::Shape::cube(), ":/volren/shaders/raycastcube.vert", ":/volren/shaders/raycastcube.frag");
}

void RaycastCube::setSize(int w, int h, int d)
{
    width = w;
    height = h;
    depth = d;
}

void RaycastCube::render(const QMatrix4x4& vp, GLenum face)
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(face);
    glEnable(GL_DEPTH_TEST);

    painter.paint("mvp", vp * matrix(),
                  "volDim", QVector3D(width, height, depth));

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
}

const QMatrix4x4 &RaycastCube::matrix()
{
    static QMatrix4x4 mat;
    mat.setToIdentity();
    mat.translate(0.5f, 0.5f, 0.5f);
    mat.scale(width - 1, height - 1, depth - 1);
    return mat;
}


