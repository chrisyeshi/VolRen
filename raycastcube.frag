#version 330

in vec3 vf_vert;

out vec4 o_color;

void main(void)
{
    o_color = vec4(vf_vert, 1.0);
}

