#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform sampler1D tf1d;
uniform int resolution;
uniform float segLen;

layout (location = 0) in vec2 vf_texLoc;

layout (location = 0) out vec4 o_full;
layout (location = 1) out vec4 o_back;

void main(void)
{
    int x = int(gl_FragCoord.x);
    int y = int(gl_FragCoord.y);
    // integrate from y to x with y being the beginning of the ray segment and x being the end of the ray segment
    const float baseSample = 0.01;
    int dirSteps = x - y;
    int steps = (0 == dirSteps) ? 1 : abs(dirSteps);
    int dir = dirSteps / steps;
    float stepsize = segLen / float(steps);
    vec4 full = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 back = vec4(0.0, 0.0, 0.0, 0.0);
    for (int s = 0; s < steps; ++s)
    {
        int tfIdx = y + s * dir;
        vec4 spotColor, backColor;
        // sample
        spotColor = texture(tf1d, float(tfIdx) / float(resolution - 1));
        // adjust
        spotColor.a = 1.0 - pow(1.0 - spotColor.a, stepsize / baseSample);
        spotColor.rgb *= spotColor.a;
        // weighted for front and back samples
        float weightBack = (steps - 1 == 0) ? 0.0 : (float(s) / float(steps - 1));
        backColor = spotColor * weightBack;
        // attenuate back
        back += backColor * (1.0 - full.a);
        full += spotColor * (1.0 - full.a);
    }
    // output
    o_full = full;
    o_back = back;
}

