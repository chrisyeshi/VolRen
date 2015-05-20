#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform sampler2D tex;

layout (location = 0) in vec2 vf_texLoc;

layout (location = 0) out vec4 o_color;

void main(void)
{
    o_color = texture(tex, vf_texLoc);
}

