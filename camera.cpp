#include "camera.h"
#include <iostream>
#include <QVector2D>
#include <QtMath>

Camera::Camera()
  : m_eye(0.f, 0.f, 4.f),
    m_center(0.f, 0.f, 0.f),
    m_up(0.f, 1.f, 0.f),
    m_zoom(initZoom),
    m_mode(PM_Perspective),
    m_aspect(1.f)
{
}

void Camera::orbit(const QVector2D &dir)
{
    QVector3D u = (m_center - m_eye).normalized();
    QVector3D s = QVector3D::crossProduct(u, m_up).normalized();
    QVector3D t = QVector3D::crossProduct(s, u).normalized();
    QVector3D rotateDir = (dir.x() * s + dir.y() * t).normalized();
    QVector3D rotateAxis = QVector3D::crossProduct(rotateDir, u).normalized();
    float angle = dir.length() * 360.f;
    QMatrix4x4 matRotate;
    matRotate.rotate(-angle, rotateAxis);
    QVector3D view = matRotate * (m_eye - m_center);
    m_eye = m_center + view;
    m_up = QVector3D::crossProduct(view, s).normalized();
}

void Camera::track(const QVector2D &dir)
{
    QVector3D u = (m_center - m_eye).normalized();
    QVector3D s = QVector3D::crossProduct(u, m_up).normalized();
    QVector3D t = QVector3D::crossProduct(s, u).normalized();
    QVector2D scaleDir = dir * focalPlaneSize();
    QVector3D trackDir = (scaleDir.x() * s + scaleDir.y() * t);
    m_eye = m_eye - trackDir;
    m_center = m_center - trackDir;
}

void Camera::zoom(const float factor)
{
    m_zoom += factor / 1000.f;
}

void Camera::reset(const QVector3D &bmin, const QVector3D &bmax)
{
    // reset the camera to at the z directon
    // reset the object to the center of the screen
    // calculate the distance from center so that everything fits in the screen
    m_center = (bmin + bmax) / 2.f;
    m_up = QVector3D(0.f, 1.f, 0.f);
    m_zoom = initZoom;
    float diagonal = (bmax - bmin).length();
    float distance = (diagonal / 2.f) / tan(fov() / 180.f * M_PI / 2.f);
    distance = distance + bmax.z() - m_center.z();
    m_eye = m_center - QVector3D(0.f, 0.f, 1.f) * distance;
    m_near = distance - diagonal / 2.f;
    m_far = distance + diagonal / 2.f;
}

const QMatrix4x4& Camera::matView() const
{
    static QMatrix4x4 mat;
    mat.setToIdentity();
    mat.lookAt(m_eye, m_center, m_up);
    return mat;
}

const QMatrix4x4& Camera::matProj() const
{
    static QMatrix4x4 mat;
    mat.setToIdentity();
    if (PM_Perspective == m_mode)
    {
        mat.perspective(fov(), m_aspect, m_near, m_far);
    } else
    {
        QVector2D size = focalPlaneSize() / 2.f;
        mat.ortho(-size.x(), size.x(), -size.y(), size.y(), m_near, m_far);
    }
    return mat;
}

//
//
//
// helper functions
//
//
//

float Camera::fov() const
{
    return (qAtan(m_zoom) / M_PI + 0.5f) * 180.f;
}

QVector2D Camera::focalPlaneSize() const
{
    float viewLen = (m_center - m_eye).length();
    float height = qTan(fov() / 2.f / 180.f * M_PI) * viewLen;
    return QVector2D(height * m_aspect * 2.f, height * 2.f);
}
