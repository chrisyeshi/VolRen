#version 330

uniform int progress;
uniform int pgrLevel;

in vec3 vf_vert;

out vec4 o_color;

void main(void)
{
    int fragY = int(gl_FragCoord.y);
    if (progress != mod(fragY, int(pow(4.0, float(pgrLevel)))))
        discard;
    o_color = vec4(vf_vert, 1.0);
}

