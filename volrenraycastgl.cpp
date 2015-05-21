#include "volrenraycastgl.h"
#include <map>
#include <QMatrix4x4>
#include <QOpenGLFunctions>
#include "imagetex.h"

namespace yy {
namespace volren {

VolRenRaycastGL::VolRenRaycastGL()
 : VolRenRaycast(Method_Raycast_GL)
{

}

VolRenRaycastGL::~VolRenRaycastGL()
{

}

std::shared_ptr<ImageAbstract> VolRenRaycastGL::output() const
{
    return std::make_shared<ImageTex>(*outTex);
}

void VolRenRaycastGL::newFBOs(int w, int h)
{
    VolRenRaycast::newFBOs(w, h);
    newFBO(w, h, &outFBO, &outTex, &outRen);
}

void VolRenRaycastGL::raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&)
{
    QOpenGLFunctions f(QOpenGLContext::currentContext());
    f.glBindFramebuffer(GL_FRAMEBUFFER, *outFBO);
    f.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f.glEnable(GL_TEXTURE_2D);
    f.glEnable(GL_TEXTURE_3D);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, *entryTex);
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, *exitTex);
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, volTex->textureId());
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, tfTex->textureId());
    painter.paint("texEntry", 0,
                  "texExit", 1,
                  "texVolume", 2,
                  "texTF", 3,
                  "volSize", QVector3D(vol()->w(), vol()->h(), vol()->d()),
                  "stepSize", stepsize,
                  "scalarMin", scalarMin,
                  "scalarMax", scalarMax);
    f.glActiveTexture(GL_TEXTURE3);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE2);
    f.glBindTexture(GL_TEXTURE_3D, 0);
    f.glActiveTexture(GL_TEXTURE1);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glActiveTexture(GL_TEXTURE0);
    f.glBindTexture(GL_TEXTURE_2D, 0);
    f.glDisable(GL_TEXTURE_3D);
    f.glDisable(GL_TEXTURE_2D);
    f.glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

} // namespace volren
} // namespace yy
