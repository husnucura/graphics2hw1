#version 330 core

uniform sampler2D ourTexture;

in vec2 TexCoord;

out vec4 fragColor;

void main(void)
{
	fragColor = vec4(vec3(texture(ourTexture, TexCoord)),1);
}
