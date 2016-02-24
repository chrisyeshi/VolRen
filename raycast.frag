#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform sampler2D texEntry;
uniform sampler2D texExit;
uniform sampler3D texVolume;
uniform sampler2D texTFFull;
uniform sampler2D texTFBack;
uniform int volFilter;
uniform vec3 volSize;
uniform float stepSize;
uniform float scalarMin;
uniform float scalarMax;
uniform int nLights;
uniform struct Light {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
} lights[10];

layout (location = 0) in vec2 vf_texLoc;

layout (location = 0) out vec4 o_color;

vec3 invVolSize;

vec3 makeGradient(vec3 spot)
{
    vec3 gradient;
    gradient.x = 0.5 * (texture(texVolume, spot + vec3(1.f/volSize.x, 0.f, 0.f)).r
                      - texture(texVolume, spot - vec3(1.f/volSize.x, 0.f, 0.f)).r);
    gradient.y = 0.5 * (texture(texVolume, spot + vec3(0.f, 1.f/volSize.y, 0.f)).r
                      - texture(texVolume, spot - vec3(0.f, 1.f/volSize.y, 0.f)).r);
    gradient.z = 0.5 * (texture(texVolume, spot + vec3(0.f, 0.f, 1.f/volSize.z)).r
                      - texture(texVolume, spot - vec3(0.f, 0.f, 1.f/volSize.z)).r);
    return gradient;
}

vec3 entryGradient(vec2 viewSpot)
{
    float delta = 1.0 / 80000.f;
    vec3 left = texture(texEntry, viewSpot + vec2(-delta, 0.0)).xyz;
    vec3 right = texture(texEntry, viewSpot + vec2(delta, 0.0)).xyz;
    vec3 top = texture(texEntry, viewSpot + vec2(0.0, delta)).xyz;
    vec3 bottom = texture(texEntry, viewSpot + vec2(0.0, -delta)).xyz;
    return cross(top - bottom, right - left);
}

vec4 getLightFactor(vec3 grad, vec3 view)
{
    if (nLights == 0)
        return vec4(1.0, 1.0, 1.0, 1.0);
    vec3 V = normalize(-view);
    vec3 N = normalize(-grad);
    vec4 acc = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < nLights; ++i)
    {
        vec3 kd = lights[i].diffuse;
        vec3 ka = lights[i].ambient;
        vec3 ks = lights[i].specular;
        float shininess = lights[i].shininess;
        vec3 L = normalize(-lights[i].direction);
        vec3 R = normalize(-reflect(L, N));
        vec3 diffuse = kd * max(dot(L, N), 0.f);
        vec3 specular = ks * pow(max(dot(R, V), 0.f), shininess);
        vec3 cf = ka + diffuse + specular;
        float af = 1.f;
        acc += vec4(cf, af);
    }
    return vec4(acc.rgb, 1.0);
}

float sampleTricubic(vec3 texCoord)
{
    vec3 x = texCoord * volSize;
    vec3 i = floor(x - vec3(0.5)) + vec3(0.5);
    vec3 a = x - i;

    vec3 a2 = a * a;
    vec3 a3 = a2 * a;

    // B-spline weighting function
    float one6th = 1.0 / 6.0;
    vec3 w0 = one6th * (-a3 + 3.0 * a2 - 3.0 * a + 1.0);
    vec3 w1 = one6th * (3.0 * a3 - 6.0 * a2 + 4.0);
    vec3 w2 = one6th * (-3.0 * a3 + 3.0 * a2 + 3.0 * a + 1.0);
    vec3 w3 = one6th * a3;

    vec3 s0 = w0 + w1;
    vec3 s1 = w2 + w3;

    vec3 t0 = i - vec3(1.0) + w1 / s0;
    vec3 t1 = i + vec3(1.0) + w3 / s1;

    t0 *= invVolSize;
    t1 *= invVolSize;

    return (texture(texVolume, vec3(t0.x, t0.y, t0.z)).r * s0.x * s0.y * s0.z
          + texture(texVolume, vec3(t1.x, t0.y, t0.z)).r * s1.x * s0.y * s0.z
          + texture(texVolume, vec3(t0.x, t1.y, t0.z)).r * s0.x * s1.y * s0.z
          + texture(texVolume, vec3(t1.x, t1.y, t0.z)).r * s1.x * s1.y * s0.z
          + texture(texVolume, vec3(t0.x, t0.y, t1.z)).r * s0.x * s0.y * s1.z
          + texture(texVolume, vec3(t1.x, t0.y, t1.z)).r * s1.x * s0.y * s1.z
          + texture(texVolume, vec3(t0.x, t1.y, t1.z)).r * s0.x * s1.y * s1.z
          + texture(texVolume, vec3(t1.x, t1.y, t1.z)).r * s1.x * s1.y * s1.z);
}

float sampleVolume(vec3 loc)
{
    if (volFilter == 2)
    { // tricubic interpolation
        return sampleTricubic(loc);
    } else
    { // trilinear or nearest
        return texture(texVolume, loc).r;
    }
}

void main(void)
{
    invVolSize = vec3(1.0) / volSize;
    vec3 entry = texture(texEntry, vf_texLoc).xyz;
    vec3 exit = texture(texExit, vf_texLoc).xyz;
    vec3 dir = normalize(exit - entry);
    float baseSample = 0.01;
    float maxLength = length(exit - entry);
    if (maxLength < stepSize)
        discard;
    vec2 scalar = vec2(0.0, 0.0); // a segment of the ray, X as the scalar value at the end of the segment, and Y as the scalar value at the beginning of the segment.
    scalar.y = sampleVolume(entry);
    scalar.y = clamp((scalar.y - scalarMin) / (scalarMax - scalarMin), 0.0, 1.0);
    vec3 spotCurr;
    vec4 lfPrev = getLightFactor(entryGradient(vf_texLoc), dir);
    vec4 lfCurr;
    vec4 acc = vec4(0.0);
    for (int step = 1; step * stepSize < maxLength; ++step)
    {
        vec3 spotCurr = entry + dir * (step * stepSize);
        scalar.x = sampleVolume(spotCurr);
        scalar.x = clamp((scalar.x - scalarMin) / (scalarMax - scalarMin), 0.0, 1.0);
        vec4 colorFull = texture(texTFFull, scalar);
        vec4 colorBack = texture(texTFBack, scalar);
        vec4 colorFront = colorFull - colorBack;
        lfCurr = getLightFactor(makeGradient(spotCurr), dir);
        acc += (colorBack * lfCurr + colorFront * lfPrev) * (1.0 - acc.a);
        if (acc.a > 0.999)
            break;
        scalar.y = scalar.x;
        lfPrev = lfCurr;
    }
    o_color = acc;
}

