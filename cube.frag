#version 330

in vec4 vf_vert;

out vec4 o_color;

void main(void)
{
    o_color = vec4(vf_vert.xyz, 1.0);
}

