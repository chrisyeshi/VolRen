#version 330
#extension GL_ARB_separate_shader_objects : enable

uniform sampler1D texTF;
uniform sampler3D texRAF;
uniform int layers;

layout (location = 0) in vec2 vf_texLoc;

layout (location = 0) out vec4 o_color;

void main(void)
{
//     int iLayer = 3;
//     float layerTexCoord = (float(iLayer) + 0.5) / float(layers);
//     vec3 layerColor = texture(texTF, layerTexCoord).rgb;
//     float layerAtten = texture(texRAF, vec3(vf_texLoc, layerTexCoord)).r;
//     o_color = vec4(vec3(layerAtten), 1.0);
//     return;

	vec3 acc = vec3(0.0);
	for (int iLayer = 0; iLayer < layers; ++iLayer)
	{
        float layerTexCoord = (float(iLayer) + 0.5) / float(layers);
        vec3 layerColor = texture(texTF, layerTexCoord).rgb;
		float layerAtten = texture(texRAF, vec3(vf_texLoc, layerTexCoord)).r;
		acc += layerColor * layerAtten;
	}
    if (acc == vec3(0.0))
        o_color = vec4(0.0);
    else
        o_color = vec4(acc, 1.0);
}

