#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <GL/glew.h>
// #include <OpenGL/gl3.h>   // The GL Header File
#include <GLFW/glfw3.h> // The GLFW header
#include <glm/glm.hpp>	// GL Math library header
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <map>
#define BUFFER_OFFSET(i) ((char *)NULL + (i))

using namespace std;

GLuint vao[3];
GLuint gProgram[3];
int gWidth, gHeight;

GLint modelingMatrixLoc[3];
GLint viewingMatrixLoc[3];
GLint projectionMatrixLoc[3];
GLint eyePosLoc[3];

glm::mat4 projectionMatrix;
glm::mat4 viewingMatrix;
glm::mat4 modelingMatrix;
glm::vec3 eyePos(0, 0, 0);
glm::vec3 eyeGaze(0, 0, -1);
glm::vec3 eyeUp(0, 1, 0);
glm::mat4 curveModelingMatrix = glm::mat4(1.0);
const int numCurveSamplePoints = 200;
unsigned int currentCurvePoint = 0;

GLfloat binomial(GLuint n, GLuint k);
float randomFloat(float min, float max)
{
	return min + ((float)std::rand()) / RAND_MAX * (max - min);
};
struct BezierCurve
{
	glm::vec3 controlPoints[4];
	vector<glm::vec3> sampledPoints;

	BezierCurve() : sampledPoints(numCurveSamplePoints) {}
	void CreateRandomControlPointAt(int i)
	{
		controlPoints[i].x = randomFloat(-1, 1);
		controlPoints[i].y = randomFloat(-1, 1);
		controlPoints[i].z = randomFloat(-7, -5);
	}
	glm::vec3 tangentAt(int index)
	{
		float t = (float)index / numCurveSamplePoints;
		float u = 1.0f - t;

		glm::vec3 tmp = 3.0f * u * u * (controlPoints[1] - controlPoints[0]) +
						6.0f * u * t * (controlPoints[2] - controlPoints[1]) +
						3.0f * t * t * (controlPoints[3] - controlPoints[2]);
		return glm::normalize(tmp);
	}
};
BezierCurve bezierCurve;

bool wireframe = false;
bool pathVisible = true;
bool freezed = false;
bool SampleSurface(const string &filename, int uSteps, int vSteps, int objId);

struct Vertex
{
	Vertex(GLfloat inX, GLfloat inY, GLfloat inZ) : x(inX), y(inY), z(inZ) {}
	GLfloat x, y, z;
	static Vertex lerp(const Vertex &v1, const Vertex &v2, GLfloat t)
	{
		return Vertex(
			(1 - t) * v1.x + t * v2.x,
			(1 - t) * v1.y + t * v2.y,
			(1 - t) * v1.z + t * v2.z);
	}
	bool operator<(const Vertex &other) const
	{

		return std::tie(x, y, z) < std::tie(other.x, other.y, other.z);
	}
	Vertex operator+(const Vertex &other) const
	{
		return Vertex(x + other.x, y + other.y, z + other.z);
	}
	Vertex operator-(const Vertex &other) const
	{
		return Vertex(x - other.x, y - other.y, z - other.z);
	}

	Vertex cross(const Vertex &other) const
	{
		return Vertex(
			y * other.z - z * other.y,
			z * other.x - x * other.z,
			x * other.y - y * other.x);
	}
	Vertex operator*(GLfloat scalar) const
	{
		return Vertex(x * scalar, y * scalar, z * scalar);
	}

	Vertex normalize() const
	{
		GLfloat length = std::sqrt(x * x + y * y + z * z);
		if (length == 0)
			return *this;
		return Vertex(x / length, y / length, z / length);
	}
};

struct Texture
{
	Texture(GLfloat inU, GLfloat inV) : u(inU), v(inV) {}
	GLfloat u, v;
};

struct Normal
{
	Normal(GLfloat inX, GLfloat inY, GLfloat inZ) : x(inX), y(inY), z(inZ) {}
	GLfloat x, y, z;
	Normal operator+(const Normal &other) const
	{
		return Normal(x + other.x, y + other.y, z + other.z);
	}

	Normal &operator+=(const Normal &other)
	{
		x += other.x;
		y += other.y;
		z += other.z;
		return *this;
	}

	// Normalize.
	Normal normalize() const
	{
		GLfloat len = std::sqrt(x * x + y * y + z * z);
		if (len == 0)
			return *this;
		return Normal(x / len, y / len, z / len);
	}
};

struct Face
{
	Face(int v[], int t[], int n[])
	{
		vIndex[0] = v[0];
		vIndex[1] = v[1];
		vIndex[2] = v[2];
		tIndex[0] = t[0];
		tIndex[1] = t[1];
		tIndex[2] = t[2];
		nIndex[0] = n[0];
		nIndex[1] = n[1];
		nIndex[2] = n[2];
	}
	GLuint vIndex[3], tIndex[3], nIndex[3];
};

vector<Vertex> gVertices[3];
vector<Texture> gTextures[3];
vector<Normal> gNormals[3];
vector<Face> gFaces[3];

GLuint gVertexAttribBuffer[3], gIndexBuffer[3];
GLint gInVertexLoc[3], gInNormalLoc[3];
int gVertexDataSizeInBytes[3], gNormalDataSizeInBytes[3], gTextureDataSizeInBytes[3];

bool ParseObj(const string &fileName, int objId)
{
	fstream myfile;

	// Open the input
	myfile.open(fileName.c_str(), std::ios::in);

	if (myfile.is_open())
	{
		string curLine;

		while (getline(myfile, curLine))
		{
			stringstream str(curLine);
			GLfloat c1, c2, c3;
			GLuint index[9];
			string tmp;

			if (curLine.length() >= 2)
			{
				if (curLine[0] == 'v')
				{
					if (curLine[1] == 't') // texture
					{
						str >> tmp; // consume "vt"
						str >> c1 >> c2;
						gTextures[objId].push_back(Texture(c1, c2));
					}
					else if (curLine[1] == 'n') // normal
					{
						str >> tmp; // consume "vn"
						str >> c1 >> c2 >> c3;
						gNormals[objId].push_back(Normal(c1, c2, c3));
					}
					else // vertex
					{
						str >> tmp; // consume "v"
						str >> c1 >> c2 >> c3;
						gVertices[objId].push_back(Vertex(c1, c2, c3));
					}
				}
				else if (curLine[0] == 'f') // face
				{
					str >> tmp; // consume "f"
					char c;
					int vIndex[3], nIndex[3], tIndex[3];
					str >> vIndex[0];
					str >> c >> c; // consume "//"
					str >> nIndex[0];
					str >> vIndex[1];
					str >> c >> c; // consume "//"
					str >> nIndex[1];
					str >> vIndex[2];
					str >> c >> c; // consume "//"
					str >> nIndex[2];

					assert(vIndex[0] == nIndex[0] &&
						   vIndex[1] == nIndex[1] &&
						   vIndex[2] == nIndex[2]); // a limitation for now

					// make indices start from 0
					for (int c = 0; c < 3; ++c)
					{
						vIndex[c] -= 1;
						nIndex[c] -= 1;
						tIndex[c] -= 1;
					}

					gFaces[objId].push_back(Face(vIndex, tIndex, nIndex));
				}
				else
				{
					cout << "Ignoring unidentified line in obj file: " << curLine << endl;
				}
			}

			// data += curLine;
			if (!myfile.eof())
			{
				// data += "\n";
			}
		}

		myfile.close();
	}
	else
	{
		return false;
	}

	assert(gVertices[objId].size() == gNormals[objId].size());

	return true;
}

bool ReadDataFromFile(
	const string &fileName, ///< [in]  Name of the shader file
	string &data)			///< [out] The contents of the file
{
	fstream myfile;

	// Open the input
	myfile.open(fileName.c_str(), std::ios::in);

	if (myfile.is_open())
	{
		string curLine;

		while (getline(myfile, curLine))
		{
			data += curLine;
			if (!myfile.eof())
			{
				data += "\n";
			}
		}

		myfile.close();
	}
	else
	{
		return false;
	}

	return true;
}

GLuint createVS(const char *shaderName)
{
	string shaderSource;

	string filename(shaderName);
	if (!ReadDataFromFile(filename, shaderSource))
	{
		cout << "Cannot find file name: " + filename << endl;
		exit(-1);
	}

	GLint length = shaderSource.length();
	const GLchar *shader = (const GLchar *)shaderSource.c_str();

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &shader, &length);
	glCompileShader(vs);

	char output[1024] = {0};
	glGetShaderInfoLog(vs, 1024, &length, output);
	printf("VS compile log: %s\n", output);

	return vs;
}

GLuint createFS(const char *shaderName)
{
	string shaderSource;

	string filename(shaderName);
	if (!ReadDataFromFile(filename, shaderSource))
	{
		cout << "Cannot find file name: " + filename << endl;
		exit(-1);
	}

	GLint length = shaderSource.length();
	const GLchar *shader = (const GLchar *)shaderSource.c_str();

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &shader, &length);
	glCompileShader(fs);

	char output[1024] = {0};
	glGetShaderInfoLog(fs, 1024, &length, output);
	printf("FS compile log: %s\n", output);

	return fs;
}

void initTexture()
{
	unsigned int texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	// set the texture wrapping/filtering options (on the currently bound texture object)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// load the texture
	int width, height, nrChannels;
	unsigned char *data = stbi_load("haunted_library.jpg", &width, &height, &nrChannels, 0);
	if (data)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else
	{
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);
}

void initShaders()
{
	// Create the programs

	gProgram[0] = glCreateProgram(); // for armadillo
	gProgram[1] = glCreateProgram(); // for background quad
	gProgram[2] = glCreateProgram(); // line

	// Create the shaders for both programs

	// for armadillo
	GLuint vs1 = createVS("vert2.glsl"); // or vert2.glsl
	GLuint fs1 = createFS("frag2.glsl"); // or frag2.glsl

	// for background quad
	GLuint vs2 = createVS("vert_quad.glsl");
	GLuint fs2 = createFS("frag_quad.glsl");

	GLuint vs3 = createVS("vert.glsl"); // or vert2.glsl
	GLuint fs3 = createFS("frag.glsl"); // or frag2.glsl

	// Attach the shaders to the programs

	glAttachShader(gProgram[0], vs1);
	glAttachShader(gProgram[0], fs1);

	glAttachShader(gProgram[1], vs2);
	glAttachShader(gProgram[1], fs2);

	glAttachShader(gProgram[2], vs3);
	glAttachShader(gProgram[2], fs3);

	// Link the programs

	glLinkProgram(gProgram[0]);
	GLint status;
	glGetProgramiv(gProgram[0], GL_LINK_STATUS, &status);

	if (status != GL_TRUE)
	{
		cout << "Program link failed" << endl;
		exit(-1);
	}

	glLinkProgram(gProgram[1]);
	glGetProgramiv(gProgram[1], GL_LINK_STATUS, &status);

	if (status != GL_TRUE)
	{
		cout << "Program link failed" << endl;
		exit(-1);
	}

	glLinkProgram(gProgram[2]);
	glGetProgramiv(gProgram[2], GL_LINK_STATUS, &status);

	if (status != GL_TRUE)
	{
		cout << "Program link failed" << endl;
		exit(-1);
	}

	// Get the locations of the uniform variables from both programs

	for (int i = 0; i < 3; ++i)
	{
		glUseProgram(gProgram[i]);

		modelingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "modelingMatrix");
		viewingMatrixLoc[i] = glGetUniformLocation(gProgram[i], "viewingMatrix");
		projectionMatrixLoc[i] = glGetUniformLocation(gProgram[i], "projectionMatrix");
		eyePosLoc[i] = glGetUniformLocation(gProgram[i], "eyePos");
	}
}

void updateOrInitVBO(int t, bool IsFirst = false)
{
	if (IsFirst)
	{
		// First-time initialization
		glGenVertexArrays(1, &vao[t]);
		assert(vao[t] > 0);

		glBindVertexArray(vao[t]);
		cout << "vao = " << vao[t] << endl;

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		assert(glGetError() == GL_NONE);

		glGenBuffers(1, &gVertexAttribBuffer[t]);
		glGenBuffers(1, &gIndexBuffer[t]);

		assert(gVertexAttribBuffer[t] > 0 && gIndexBuffer[t] > 0);

		glBindBuffer(GL_ARRAY_BUFFER, gVertexAttribBuffer[t]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIndexBuffer[t]);
	}
	else
	{
		// Update existing VBO
		glBindVertexArray(vao[t]);
		glBindBuffer(GL_ARRAY_BUFFER, gVertexAttribBuffer[t]);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gIndexBuffer[t]);
	}

	// Calculate the data sizes
	gVertexDataSizeInBytes[t] = gVertices[t].size() * 3 * sizeof(GLfloat);
	gNormalDataSizeInBytes[t] = gNormals[t].size() * 3 * sizeof(GLfloat);
	gTextureDataSizeInBytes[t] = gTextures[t].size() * 2 * sizeof(GLfloat);
	int indexDataSizeInBytes = gFaces[t].size() * 3 * sizeof(GLuint);

	GLfloat *vertexData = new GLfloat[gVertices[t].size() * 3];
	GLfloat *normalData = new GLfloat[gNormals[t].size() * 3];
	GLfloat *textureData = new GLfloat[gTextures[t].size() * 2];
	GLuint *indexData = new GLuint[gFaces[t].size() * 3];

	float minX = 1e6, maxX = -1e6;
	float minY = 1e6, maxY = -1e6;
	float minZ = 1e6, maxZ = -1e6;

	// Prepare vertex data
	for (int i = 0; i < gVertices[t].size(); ++i)
	{
		vertexData[3 * i] = gVertices[t][i].x;
		vertexData[3 * i + 1] = gVertices[t][i].y;
		vertexData[3 * i + 2] = gVertices[t][i].z;

		minX = std::min(minX, gVertices[t][i].x);
		maxX = std::max(maxX, gVertices[t][i].x);
		minY = std::min(minY, gVertices[t][i].y);
		maxY = std::max(maxY, gVertices[t][i].y);
		minZ = std::min(minZ, gVertices[t][i].z);
		maxZ = std::max(maxZ, gVertices[t][i].z);
	}

	std::cout << "minX = " << minX << std::endl;
	std::cout << "maxX = " << maxX << std::endl;
	std::cout << "minY = " << minY << std::endl;
	std::cout << "maxY = " << maxY << std::endl;
	std::cout << "minZ = " << minZ << std::endl;
	std::cout << "maxZ = " << maxZ << std::endl;

	// Prepare normal data
	for (int i = 0; i < gNormals[t].size(); ++i)
	{
		normalData[3 * i] = gNormals[t][i].x;
		normalData[3 * i + 1] = gNormals[t][i].y;
		normalData[3 * i + 2] = gNormals[t][i].z;
	}

	// Prepare texture data
	for (int i = 0; i < gTextures[t].size(); ++i)
	{
		textureData[2 * i] = gTextures[t][i].u;
		textureData[2 * i + 1] = gTextures[t][i].v;
	}

	// Prepare index data
	for (int i = 0; i < gFaces[t].size(); ++i)
	{
		indexData[3 * i] = gFaces[t][i].vIndex[0];
		indexData[3 * i + 1] = gFaces[t][i].vIndex[1];
		indexData[3 * i + 2] = gFaces[t][i].vIndex[2];
	}

	glBufferData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes[t] + gNormalDataSizeInBytes[t] + gTextureDataSizeInBytes[t], 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, gVertexDataSizeInBytes[t], vertexData);
	glBufferSubData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes[t], gNormalDataSizeInBytes[t], normalData);
	glBufferSubData(GL_ARRAY_BUFFER, gVertexDataSizeInBytes[t] + gNormalDataSizeInBytes[t], gTextureDataSizeInBytes[t], textureData);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexDataSizeInBytes, indexData, GL_STATIC_DRAW);

	// Done copying; can free now
	delete[] vertexData;
	delete[] normalData;
	delete[] textureData;
	delete[] indexData;

	// Set the vertex attribute pointers
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes[t]));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(gVertexDataSizeInBytes[t] + gNormalDataSizeInBytes[t]));
}

void initVBO()
{
	for (size_t t = 0; t < 3; t++)
	{
		updateOrInitVBO(t, true);
	}
}
float bernsteinPol(int i, int n, float t)
{
	return binomial(n, i) * std::pow(t, i) * std::pow(1 - t, n - i);
}
void sampleBezierCurve(bool isFirst = false)
{
	int n = 3;
	gVertices[2].clear();
	bezierCurve.sampledPoints[0] = bezierCurve.controlPoints[0];
	bezierCurve.sampledPoints[numCurveSamplePoints - 1] = bezierCurve.controlPoints[3];

	for (int i = 1; i + 1 < numCurveSamplePoints; ++i)
	{
		float t = float(i) / (numCurveSamplePoints - 1);
		glm::vec3 p(0.0f);
		for (int j = 0; j <= n; ++j)
		{
			float B = bernsteinPol(j, n, t);
			p += B * bezierCurve.controlPoints[j];
		}

		bezierCurve.sampledPoints[i] = p;
		gVertices[2].push_back(Vertex(p.x, p.y, p.z));
	}
	updateOrInitVBO(2, isFirst);
}

void initBezierCurve()
{
	for (int i = 0; i < 4; i++)
	{
		bezierCurve.CreateRandomControlPointAt(i);
	}
	sampleBezierCurve(true);
}
void init(int uSteps, int vSteps)
{
	// ParseObj("armadillo.obj", 0);
	SampleSurface("bezier.txt", uSteps, vSteps, 0);
	ParseObj("quad.obj", 1);

	glEnable(GL_DEPTH_TEST);
	initTexture();
	initShaders();
	initVBO();
	initBezierCurve();
}

void drawScene()
{
	for (size_t t = 0; t < 2; t++)
	{
		if (t == 0)
		{
			if (wireframe)
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			}
			else
			{
				glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			}
		}
		else
		{
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		}

		// Set the active program and the values of its uniform variables
		glUseProgram(gProgram[t]);
		glUniformMatrix4fv(projectionMatrixLoc[t], 1, GL_FALSE, glm::value_ptr(projectionMatrix));
		glUniformMatrix4fv(viewingMatrixLoc[t], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
		glUniformMatrix4fv(modelingMatrixLoc[t], 1, GL_FALSE, glm::value_ptr(modelingMatrix));
		glUniform3fv(eyePosLoc[t], 1, glm::value_ptr(eyePos));

		glBindVertexArray(vao[t]);

		if (t == 1)
			glDepthMask(GL_FALSE);

		glDrawElements(GL_TRIANGLES, gFaces[t].size() * 3, GL_UNSIGNED_INT, 0);

		if (t == 1)
			glDepthMask(GL_TRUE);
	}
	if (!pathVisible)
		return;
	glUseProgram(gProgram[2]);
	glUniformMatrix4fv(projectionMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(projectionMatrix));
	glUniformMatrix4fv(viewingMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(viewingMatrix));
	glUniformMatrix4fv(modelingMatrixLoc[2], 1, GL_FALSE, glm::value_ptr(curveModelingMatrix));
	glUniform3fv(eyePosLoc[2], 1, glm::value_ptr(eyePos));

	glBindVertexArray(vao[2]);

	glDrawArrays(GL_LINE_STRIP, 0, gVertices[2].size());
}
void ResampleCurve()
{
	glm::vec3 p3 = bezierCurve.controlPoints[2];
	glm::vec3 p4 = bezierCurve.controlPoints[3];

	bezierCurve.controlPoints[0] = p4;
	bezierCurve.controlPoints[1] = 2.0f * p4 - p3;

	for (int i = 2; i < 4; i++)
	{
		bezierCurve.CreateRandomControlPointAt(i);
	}

	sampleBezierCurve();
}
glm::vec3 previousUp = eyeUp;
glm::mat4
calculateRotationMatrix()
{
	glm::vec3 forward = glm::normalize(bezierCurve.tangentAt(currentCurvePoint));
	glm::vec3 right = glm::normalize(glm::cross(forward, previousUp));
	glm::vec3 up = glm::normalize(glm::cross(right, forward));

	if (!freezed)
		previousUp = up;

	glm::mat3 basis(right, up, forward);
	return glm::mat4(basis);
}

void display()
{
	glClearColor(0, 0, 0, 1);
	glClearDepth(1.0f);
	glClearStencil(0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
	glm::vec3 position = bezierCurve.sampledPoints[currentCurvePoint];
	glm::mat4 rotation = calculateRotationMatrix();
	glm::mat4 translation = glm::translate(glm::mat4(1.0f), position);
	modelingMatrix = rotation;
	static float rollDeg = 0;
	static float changeRoll = 2;
	float rollRad = (float)(rollDeg / 180.f) * M_PI;
	if (!freezed)
		rollDeg += changeRoll;
	if (rollDeg >= 25.f || rollDeg <= -25.f)
	{
		changeRoll *= -1.f;
	}

	glm::quat q0(0, 0, 0, 1);
	glm::quat q1(0, 0, 1, 0);
	glm::quat q = glm::mix(q0, q1, 0.0f);

	float sint = sin(rollRad / 2);
	glm::quat rollQuat(cos(rollRad / 2), sint * q.x, sint * q.y, sint * q.z);
	modelingMatrix = glm::toMat4(rollQuat) * modelingMatrix;
	modelingMatrix = translation * modelingMatrix;

	drawScene();

	if (freezed)
		return;
	currentCurvePoint++;
	if (currentCurvePoint >= numCurveSamplePoints)
	{
		currentCurvePoint = 0;
		ResampleCurve();
	}
}
void reshape(GLFWwindow *window, int w, int h)
{
	w = w < 1 ? 1 : w;
	h = h < 1 ? 1 : h;

	gWidth = w;
	gHeight = h;

	glViewport(0, 0, w, h);

	// glMatrixMode(GL_PROJECTION);
	// glLoadIdentity();
	// glOrtho(-10, 10, -10, 10, -10, 10);
	// gluPerspective(45, 1, 1, 100);

	// Use perspective projection

	float fovyRad = (float)(45.0 / 180.0) * M_PI;
	projectionMatrix = glm::perspective(fovyRad, 1.0f, 1.0f, 100.0f);

	// Assume a camera position and orientation (camera is at
	// (0, 0, 0) with looking at -z direction and its up vector pointing
	// at +y direction)

	viewingMatrix = glm::lookAt(eyePos, eyePos + eyeGaze, eyeUp);

	// glMatrixMode(GL_MODELVIEW);
	// glLoadIdentity();
}

void keyboard(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_Q && action == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS)
	{
		wireframe = !wireframe;
	}
	else if (key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		pathVisible = !pathVisible;
	}
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		freezed = !freezed;
	}
	else if (key == GLFW_KEY_F && action == GLFW_PRESS)
	{
		// glShadeModel(GL_FLAT);
	}
}

void mainLoop(GLFWwindow *window)
{
	while (!glfwWindowShouldClose(window))
	{
		display();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

struct BezierSurface
{
	std::vector<std::vector<Vertex>> controlPoints;
	BezierSurface(const std::vector<std::vector<Vertex>> &initControlPoints)
	{
		controlPoints = initControlPoints;
	}
};

struct BezierSurfaceSampleParams
{
	int u;
	int v;
};

GLuint factorial(GLuint n)
{
	GLuint result = 1;
	for (GLuint i = 2; i <= n; ++i)
	{
		result *= i;
	}
	return result;
}

GLfloat binomial(GLuint n, GLuint k)
{
	assert(k <= n);
	return (GLfloat)factorial(n) / (GLfloat)(factorial(k) * factorial(n - k));
}

Vertex sampleBezierSurface(const BezierSurface &surface, float u, float v)
{
	size_t m = surface.controlPoints.size() - 1;
	size_t n = surface.controlPoints[0].size() - 1;

	Vertex result(0, 0, 0);
	for (size_t i = 0; i <= m; ++i)
	{
		for (size_t j = 0; j <= n; ++j)
		{
			GLfloat bU = binomial(m, i) * std::pow(u, i) * std::pow(1 - u, m - i);
			GLfloat bV = binomial(n, j) * std::pow(v, j) * std::pow(1 - v, n - j);
			GLfloat weight = bU * bV;
			result = result + surface.controlPoints[i][j] * weight;
		}
	}
	return result;
}

void computeVertexNormals(
	const std::vector<Vertex> &vertices,
	const std::vector<Face> &faces,
	std::vector<Normal> &outNormals)
{
	outNormals.resize(vertices.size(), Normal(0, 0, 0));
	std::vector<int> count(vertices.size(), 0);

	for (const Face &face : faces)
	{
		const Vertex &v0 = vertices[face.vIndex[0]];
		const Vertex &v1 = vertices[face.vIndex[1]];
		const Vertex &v2 = vertices[face.vIndex[2]];

		Vertex edge1 = v1 - v0;
		Vertex edge2 = v2 - v0;
		Vertex faceNormal = edge1.cross(edge2).normalize();

		for (int i = 0; i < 3; i++)
		{
			outNormals[face.vIndex[i]].x += faceNormal.x;
			outNormals[face.vIndex[i]].y += faceNormal.y;
			outNormals[face.vIndex[i]].z += faceNormal.z;
			count[face.vIndex[i]]++;
		}
	}

	// Average and normalize each vertex normal.
	for (size_t i = 0; i < outNormals.size(); i++)
	{
		if (count[i] > 0)
		{
			outNormals[i].x /= count[i];
			outNormals[i].y /= count[i];
			outNormals[i].z /= count[i];
			Normal norm(outNormals[i].x, outNormals[i].y, outNormals[i].z);
			outNormals[i] = norm.normalize();
		}
	}
}

void generateSurfaces(
	const std::vector<BezierSurface> &surfaces,
	const BezierSurfaceSampleParams &sampleParams,
	std::vector<Vertex> &gVertices,
	std::vector<Normal> &gNormals,
	std::vector<Face> &gFaces)
{
	gVertices.clear();
	gFaces.clear();

	std::map<Vertex, int> vertexMap;

	for (const auto &surface : surfaces)
	{
		std::vector<std::vector<int>> vertexIndices(sampleParams.u, std::vector<int>(sampleParams.v, -1));

		for (int i = 0; i < sampleParams.u; ++i)
		{
			float u = float(i) / (sampleParams.u - 1);
			for (int j = 0; j < sampleParams.v; ++j)
			{
				float v = (float)j / (sampleParams.v - 1);
				Vertex sample = sampleBezierSurface(surface, u, v);

				auto it = vertexMap.find(sample);
				int globalIndex;
				if (it == vertexMap.end())
				{
					globalIndex = gVertices.size();
					gVertices.push_back(sample);
					vertexMap[sample] = globalIndex;
				}
				else
				{
					globalIndex = it->second;
				}
				vertexIndices[i][j] = globalIndex;
			}
		}

		for (int i = 0; i + 1 < sampleParams.u; ++i)
		{
			for (int j = 0; j + 1 < sampleParams.v; ++j)
			{
				int idx0 = vertexIndices[i][j];
				int idx1 = vertexIndices[i + 1][j];
				int idx2 = vertexIndices[i][j + 1];
				int idx3 = vertexIndices[i + 1][j + 1];
				int t0[3] = {idx0, idx1, idx2};
				int n0[3] = {idx0, idx1, idx2};
				gFaces.push_back(Face(t0, t0, n0));
				int t1[3] = {idx1, idx3, idx2};
				int n1[3] = {idx1, idx3, idx2};
				gFaces.push_back(Face(t1, t1, n1));
			}
		}
	}

	computeVertexNormals(gVertices, gFaces, gNormals);
}

std::vector<BezierSurface> parseBezierSurfaces(const std::string &fileName)
{
	std::vector<BezierSurface> bezierSurfaces;
	std::ifstream file(fileName);

	if (!file.is_open())
	{
		std::cerr << "Error opening file: " << fileName << std::endl;
		return bezierSurfaces;
	}

	std::string line;
	std::vector<std::vector<Vertex>> surfaceVertices;
	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		std::stringstream ss(line);
		std::string vertexStr;
		std::vector<Vertex> rowVertices;

		while (std::getline(ss, vertexStr, ','))
		{
			float x, y, z;
			std::stringstream vertexStream(vertexStr);
			vertexStream >> x >> y >> z;
			rowVertices.push_back(Vertex(x, y, z));
		}
		surfaceVertices.push_back(rowVertices);
		if (surfaceVertices.size() == 4)
		{
			bezierSurfaces.push_back(BezierSurface(surfaceVertices));
			surfaceVertices.clear();
		}
	}
	file.close();
	return bezierSurfaces;
}
bool SampleSurface(const string &filename, int uSteps, int vSteps, int objId)
{
	std::vector<BezierSurface> bezierSurfaces = parseBezierSurfaces(filename);

	generateSurfaces(bezierSurfaces, {uSteps, vSteps}, gVertices[objId], gNormals[objId], gFaces[objId]);

	assert(gVertices[objId].size() == gNormals[objId].size());

	return true;
}

int main(int argc, char **argv) // Create Main Function For Bringing It All Together
{
	int samplePArameter;
	try
	{
		samplePArameter = int(atoi(argv[1]));
	}
	catch (exception &e)
	{
		cerr << "argument error" << argv[1] << "\n";
		exit(-1);
	}
	GLFWwindow *window;
	if (!glfwInit())
	{
		exit(-1);
	}

	// glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	// glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	// glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);
	// glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	int width = 640, height = 480;
	window = glfwCreateWindow(width, height, "Simple Example", NULL, NULL);

	if (!window)
	{
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// Initialize GLEW to setup the OpenGL Function pointers
	if (GLEW_OK != glewInit())
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return EXIT_FAILURE;
	}

	char rendererInfo[512] = {0};
	strcpy(rendererInfo, (const char *)glGetString(GL_RENDERER));
	strcat(rendererInfo, " - ");
	strcat(rendererInfo, (const char *)glGetString(GL_VERSION));
	glfwSetWindowTitle(window, rendererInfo);

	init(samplePArameter, samplePArameter);

	glfwSetKeyCallback(window, keyboard);
	glfwSetWindowSizeCallback(window, reshape);

	reshape(window, width, height); // need to call this once ourselves
	mainLoop(window);				// this does not return unless the window is closed

	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}