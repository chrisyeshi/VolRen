#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform vec3 volDim;
uniform int progress;
uniform int pgrLevel;

layout (location = 0) in vec3 vf_attr;

layout (location = 0) out vec4 o_color;

void main(void)
{
    int fragY = int(gl_FragCoord.y);
    if (progress != mod(fragY, int(pow(4.0, float(pgrLevel)))))
        discard;
    if (vf_attr.x < (0.5 / volDim.x) || vf_attr.x > (1.0 - 0.5 / volDim.x)
     || vf_attr.y < (0.5 / volDim.y) || vf_attr.y > (1.0 - 0.5 / volDim.y)
     || vf_attr.z < (0.5 / volDim.z) || vf_attr.z > (1.0 - 0.5 / volDim.z))
        discard;
    o_color = vec4(vf_attr, 1.0);
}

