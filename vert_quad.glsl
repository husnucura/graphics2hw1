#version 330 core

uniform mat4 modelingMatrix;
uniform mat4 viewingMatrix;
uniform mat4 projectionMatrix;

layout(location = 0) in vec3 inVertex;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexture;

out vec2 TexCoord;

void main(void)
{
	gl_Position = vec4(inVertex, 1);
	
	// interpolate the texture
	TexCoord = inTexture;
}

