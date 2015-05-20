#ifndef CAMERA_H
#define CAMERA_H

#undef near
#undef far

#include <QMatrix4x4>
#include <QVector3D>
#include <QVector2D>

class Camera
{
public:
    Camera();

    enum ProjMode { PM_Perspective, PM_Orthographic };

    void setProjMode(const ProjMode& mode) { m_mode = mode; }
    void setAspectRatio(const float ratio) { m_aspect = ratio; }
    void orbit(const QVector2D& dir);
    void track(const QVector2D& dir);
    void zoom(const float factor);
    void reset(const QVector3D& bmin, const QVector3D& bmax);
    const QMatrix4x4& matView() const;
    const QMatrix4x4& matProj() const;
    float near() const { return m_near; }
    float far() const { return m_far; }

private:
    const float initZoom = -0.5;

    QVector3D m_eye;
    QVector3D m_center;
    QVector3D m_up;
    float m_zoom;
    ProjMode m_mode;
    float m_aspect;
    float m_near, m_far;

    // helper functions
    float fov() const;
    QVector2D focalPlaneSize() const;
};

#endif // CAMERA_H
