#ifndef RAYCASTFURSTUM_H
#define RAYCASTFURSTUM_H

#include <memory>
#include <QMatrix4x4>
#include "raycastcube.h"

namespace yy {
namespace volren {

class RaycastFurstum
{
public:
    RaycastFurstum();
    ~RaycastFurstum();

    void initializeGL();
    // setResolution changes the texture pointers
    void setResolution(int w, int h);
    int getTextureWidth() const { return texWidth; }
    int getTextureHeight() const { return texHeight; }
    void setVolumeDimension(int w, int h, int d);
    const QMatrix4x4& modelMatrix() { return cube.matrix(); }
    // a getter that doesn't render a new texture
    std::shared_ptr<GLuint> entryTexture() const { return entryTex; }
    // providing view and projection matrix will render a new texture
    std::shared_ptr<GLuint> entryTexture(const QMatrix4x4& v, const QMatrix4x4& p);
    // a getter that doesn't render a new texture
    std::shared_ptr<GLuint> exitTexture() const { return exitTex; }
    // providing view and projection matrix will render a new texture
    std::shared_ptr<GLuint> exitTexture(const QMatrix4x4& v, const QMatrix4x4& p);

protected:

private:
    const int defaultFBOSize = 480;
    RaycastCube cube;
    std::shared_ptr<GLuint> entryFBO, exitFBO;
    std::shared_ptr<GLuint> entryTex, exitTex;
    std::shared_ptr<GLuint> entryRen, exitRen;
    Shape plane;
    Painter pNear, pFar;
    int texWidth, texHeight;
    int volWidth, volHeight, volDepth;

    void newFBOs();
    void newFBO(int w, int h, std::shared_ptr<GLuint>* fbo, std::shared_ptr<GLuint>* tex, std::shared_ptr<GLuint>* ren) const;

};

} // namespace volren
} // namespace yy

#endif // RAYCASTFURSTUM_H
