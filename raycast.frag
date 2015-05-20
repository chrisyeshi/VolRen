#version 330
// #extension GL_ARB_explicit_uniform_location : enable
#extension GL_ARB_separate_shader_objects : enable

// layout (location = 0) uniform sampler2D texEntry;
// layout (location = 1) uniform sampler2D texExit;
// layout (location = 2) uniform sampler3D texVolume;
// layout (location = 3) uniform sampler2D texTF;
// layout (location = 4) uniform vec3 volSize;
// layout (location = 5) uniform float stepSize;

uniform sampler2D texEntry;
uniform sampler2D texExit;
uniform sampler3D texVolume;
uniform sampler2D texTF;
uniform vec3 volSize;
uniform float stepSize;
uniform float scalarMin;
uniform float scalarMax;

layout (location = 0) in vec2 vf_texLoc;

layout (location = 0) out vec4 o_color;

void main(void)
{
    vec3 entry = texture(texEntry, vf_texLoc).xyz;
    vec3 exit = texture(texExit, vf_texLoc).xyz;
    vec3 dir = normalize(exit - entry);
    float baseSample = 0.01;
    float maxLength = length(exit - entry);
    int totalSteps = int(maxLength / stepSize);
    vec2 scalar = vec2(0.0, 0.0); // a segment of the ray, X as the scalar value at the end of the segment, and Y as the scalar value at the beginning of the segment.
    vec4 acc = vec4(0.0);
    for (int step = 0; step * stepSize < maxLength; ++step)
    {
        vec3 spot = entry + dir * (step * stepSize);
        scalar.x = texture(texVolume, spot).r;
        scalar.x = clamp((scalar.x - scalarMin) / (scalarMax - scalarMin), 0.0, 1.0);
        vec4 spotColor = texture(texTF, scalar);
        acc += spotColor * (1.0 - acc.a);
        if (acc.a > 0.999)
            break;
        scalar.y = scalar.x;
    }
    o_color = acc;
}

