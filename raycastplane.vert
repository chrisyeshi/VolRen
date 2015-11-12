#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform vec3 volDim;

layout (location = 0) in vec4 i_vert;
layout (location = 1) in vec3 i_attr;

layout (location = 0) out vec3 vf_attr;

void main(void)
{
    gl_Position = vec4(i_vert.xy * 2.0 - 1.0, 0.0, 1.0);
    vf_attr = i_attr * (volDim - 1.0) / volDim + 0.5 / volDim;
}

