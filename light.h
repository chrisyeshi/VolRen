#ifndef LIGHT_H
#define LIGHT_H

#include <QVector3D>

namespace yy {
namespace volren {

class Light
{
public:
    Light() : direction(0.f, 0.f, 1.f), color(1.f, 1.f, 1.f),
        ambient(0.1), diffuse(0.9), specular(0.2), shininess(15.f) {}

    QVector3D direction;
    QVector3D color;
    float ambient, diffuse, specular, shininess;
};

} // namespace volren
} // namespace yy

#endif // LIGHT_H

