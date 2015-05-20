#version 330
#extension GL_ARB_separate_shader_objects : enable

layout (location = 0) in vec4 i_vert;

layout (location = 0) out vec2 vf_texLoc;

void main(void)
{
    gl_Position = vec4(i_vert.xy * 2.0 - 1.0, 0.0, 1.0);
    vf_texLoc = i_vert.xy;
}

