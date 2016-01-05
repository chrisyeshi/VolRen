#ifndef LIGHT_H
#define LIGHT_H

#include <QVector3D>
#include <json/json.h>

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

    Json::Value toJson() const
    {
        Json::Value ret;
        ret["Direction"][0] = direction.x();
        ret["Direction"][1] = direction.y();
        ret["Direction"][2] = direction.z();
        ret["Color"][0] = color.x();
        ret["Color"][1] = color.y();
        ret["Color"][2] = color.z();
        ret["Ambient"] = ambient;
        ret["Diffuse"] = diffuse;
        ret["Specular"] = specular;
        ret["Shininess"] = shininess;
        return ret;
    }

    void fromJson(const Json::Value& json)
    {
        direction.setX(json["Direction"][0].asFloat());
        direction.setY(json["Direction"][1].asFloat());
        direction.setZ(json["Direction"][2].asFloat());
        color.setX(json["Color"][0].asFloat());
        color.setY(json["Color"][1].asFloat());
        color.setZ(json["Color"][2].asFloat());
        ambient = json["Ambient"].asFloat();
        diffuse = json["Diffuse"].asFloat();
        specular = json["Specular"].asFloat();
        shininess = json["Shininess"].asFloat();
    }
};

} // namespace volren
} // namespace yy

#endif // LIGHT_H

