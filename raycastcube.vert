#version 330

layout (location = 0) in vec4 vert;

uniform mat4 mvp;
uniform vec3 volDim;

out vec3 vf_vert;

void main(void)
{
    gl_Position = mvp * vert;
    vf_vert = vert.xyz * (volDim - 1.0) / volDim + 0.5 / volDim;
}
