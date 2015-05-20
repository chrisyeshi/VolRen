#version 330

layout (location = 0) in vec4 vert;

uniform mat4 mvp;

out vec4 vf_vert;

void main(void)
{
    gl_Position = mvp * vert;
    vf_vert = vert;
}

