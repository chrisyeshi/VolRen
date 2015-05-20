#ifndef VOLRENRAYCASTGL_H
#define VOLRENRAYCASTGL_H

#include "volrenraycast.h"
#include <QSharedPointer>
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <memory>
#include "volume.h"
#include "raycastcube.h"
#include "painter.h"
#include "TF.h"
#include "tfintegrater.h"
#include "imageabstract.h"

class VolRenRaycastGL : public VolRenRaycast
{
public:
    VolRenRaycastGL();
    virtual ~VolRenRaycastGL();

    virtual std::shared_ptr<ImageAbstract> output() const;

protected:
    virtual void newFBOs(int w, int h);
    virtual void raycast(const QMatrix4x4&, const QMatrix4x4&, const QMatrix4x4&);

protected:
    std::shared_ptr<GLuint> outFBO, outTex, outRen;
};

#endif // VOLRENRAYCASTGL_H
