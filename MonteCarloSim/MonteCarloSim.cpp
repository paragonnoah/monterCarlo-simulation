// MonteCarlo.cpp : Defines the entry point for the console application.

#include <string>
#include <vector>
#include "Common.h"

// OpenCL C API
#include <CL/opencl.h>

// OpenCL C++ API
#include "cl.hpp"

// GLUT
//#include <GL\glut.h> //  for windows
#include <GL/glut.h> //linux





// Visualization
#include "camera.hpp"

int windowWidth = 600;
int windowHeight = 600;

Camera camera;
bool keysPressed[256];
int method = 1;

cl_float4* visualizationBufferCPU;
cl_mem visualizationBufferGPU;

// Common
cl_platform_id platformID;
cl_device_id deviceID;
cl_context context;
cl_command_queue queue;

// Iso-surface raycasting
cl_program visualizationProgram;
cl_kernel isosurfaceRaycastingKernel;
cl_kernel alphaBlendedKernel;

float isoValue = 0.5f;
float alphaExponent = 2.0f;
float alphaCenter = 0.5f;

// Volume data
float* volumeData;
int volumeSize[3];
cl_mem volumeDataGPU;

// Scattering simulation
struct photon {
	cl_float4 origin;
	cl_float4 direction;
	cl_float energy;
};

// OpenCL program
cl_program photonProgram;

// OpenCL kernels
cl_kernel resetSimulationKernel;
cl_kernel simulationKernel;
cl_kernel visualizationKernel;

// Problem set size
size_t workGroupSize = 0;
int maxComputeUnits = 0;
size_t problemSize = 0;

// Random Generator Seed
cl_mem seedGPU;

// Photon store
cl_mem photonBufferGPU;

// Energy store
const int resolution = 64;
cl_mem simulationBufferGPU;

int iteration = 0;
cl_float4 lightSourcePosition;
void loadVolume(const char* fileName) {
	FILE* dataFile = fopen(fileName, "rb");
	char* magicNum = new char[2];
	fread(magicNum, sizeof(char), 2, dataFile);
	if ('V' == magicNum[0] && 'F' == magicNum[1]) {
		fread(volumeSize, sizeof(int), 3, dataFile);
		volumeData = new float[volumeSize[0] * volumeSize[1] * volumeSize[2]];
		fread(volumeData, sizeof(float), volumeSize[0] * volumeSize[1] * volumeSize[2], dataFile);
	}
	else {
		std::cout << "Can't open volume file %s\n" << fileName << std::endl;
	}
}

void init()
{
  // Minimal OpenCL infrastructure
	clGetPlatformIDs(1, &platformID, NULL);
	clGetDeviceIDs(platformID, CL_DEVICE_TYPE_CPU, 1, &deviceID, NULL);
	cl_context_properties contextProperties[] =
					{ CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0 };
	context = clCreateContext(contextProperties, 1, &deviceID, NULL, NULL, NULL);
	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, NULL);

	// Visualization buffers
	visualizationBufferCPU = new cl_float4[windowWidth*windowHeight];
	visualizationBufferGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) *  windowWidth * windowHeight, NULL, NULL);

	// IsoSurface raycasting
	std::string source = FileToString("../kernels/visualization.cl");
	const char* csource = source.c_str();

	visualizationProgram = clCreateProgramWithSource(context, 1, &csource, NULL, NULL);
	cl_int err = clBuildProgram(visualizationProgram, 1, &deviceID, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t logLength;
		clGetProgramBuildInfo(visualizationProgram, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLength);
		char* log = new char[logLength + 1];  // allocate +1
clGetProgramBuildInfo(visualizationProgram, deviceID, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
log[logLength] = '\0';  // 🔒 ensure null termination
std::cout << log << std::endl;
delete[] log;

		exit(-1);
	}

	isosurfaceRaycastingKernel = clCreateKernel(visualizationProgram, "isosurface", &err);
	if (!CheckCLError(err)) exit(-1);

	alphaBlendedKernel = clCreateKernel(visualizationProgram, "alphaBlended", &err);
	if (!CheckCLError(err)) exit(-1);

	loadVolume("../volumes/head.vox");
	if (NULL == volumeData) {
		exit(-1);
	}

	volumeDataGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * volumeSize[0] * volumeSize[1] * volumeSize[2], NULL, NULL);
	clEnqueueWriteBuffer(queue, volumeDataGPU, CL_TRUE, 0, sizeof(float) * volumeSize[0] * volumeSize[1] * volumeSize[2], volumeData, 0, NULL, NULL);

	// Scattering simulation
	std::string mcsource = FileToString("../kernels/programs.cl");
	const char* mccsource = mcsource.c_str();

	photonProgram = clCreateProgramWithSource(context, 1, &mccsource, NULL, NULL);
	err = clBuildProgram(photonProgram, 1, &deviceID, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		size_t logLength;
		clGetProgramBuildInfo(photonProgram, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logLength);
		char* log = new char[logLength];
		clGetProgramBuildInfo(photonProgram, deviceID, CL_PROGRAM_BUILD_LOG, logLength, log, 0);
		std::cout << log << std::endl;
		delete[] log;
		exit(-1);
	}

	resetSimulationKernel = clCreateKernel(photonProgram, "resetSimulation", &err);
	if (!CheckCLError(err)) exit(-1);
	simulationKernel = clCreateKernel(photonProgram, "simulation", &err);
	if (!CheckCLError(err)) exit(-1);
	visualizationKernel = clCreateKernel(photonProgram, "visualization", &err);
	if (!CheckCLError(err)) exit(-1);

	// working set size
	clGetKernelWorkGroupInfo(simulationKernel, deviceID, CL_KERNEL_WORK_GROUP_SIZE,
		sizeof(workGroupSize), &workGroupSize, NULL);
	clGetDeviceInfo(deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &maxComputeUnits, NULL);
	problemSize = workGroupSize * maxComputeUnits;

	std::cout << "Working set: " << workGroupSize << " * " << maxComputeUnits << " = " << problemSize << std::endl;

	// init random number generator
	cl_uint4* seedCPU = new cl_uint4[workGroupSize * maxComputeUnits];
	for (size_t i = 0; i < workGroupSize * maxComputeUnits; ++i) {
		seedCPU[i].s[0] = rand();
		seedCPU[i].s[1] = rand();
		seedCPU[i].s[2] = rand();
		seedCPU[i].s[3] = rand();
	}
	seedGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint4) * workGroupSize * maxComputeUnits, NULL, NULL);
	clEnqueueWriteBuffer(queue, seedGPU,
		CL_TRUE, 0, sizeof(cl_uint4) * workGroupSize * maxComputeUnits,
		seedCPU, 0, NULL, NULL);

	// photon buffer
	photonBufferGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(struct photon) * workGroupSize * maxComputeUnits, NULL, NULL);

	// simulation buffer
	simulationBufferGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * resolution * resolution * resolution, NULL, NULL);

	// light source parameters
	lightSourcePosition.s[0] = 0.6f;
	lightSourcePosition.s[1] = 0.5f;
	lightSourcePosition.s[2] = 0.5f;
	lightSourcePosition.s[3] = 0.0f;
}

void resetSimulation(int resolution, cl_mem simulationBufferGPU) {
	cl_int err;
	err  = clSetKernelArg(resetSimulationKernel, 0, sizeof(int), &resolution);
	err |= clSetKernelArg(resetSimulationKernel, 1, sizeof(cl_mem), &simulationBufferGPU);
	if (!CheckCLError(err)) return;

	size_t globalSize = resolution * resolution * resolution;
	clEnqueueNDRangeKernel(queue, resetSimulationKernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);
	clFinish(queue);
}



void simulationStep() {
	if (1 == iteration) {
		resetSimulation(resolution, simulationBufferGPU);
	}

	clSetKernelArg(simulationKernel, 0, sizeof(int), &iteration);
	clSetKernelArg(simulationKernel, 1, sizeof(cl_mem), &seedGPU);
	clSetKernelArg(simulationKernel, 2, sizeof(cl_mem), &photonBufferGPU);
	clSetKernelArg(simulationKernel, 3, sizeof(int), &resolution);
	clSetKernelArg(simulationKernel, 4, sizeof(cl_mem), &simulationBufferGPU);
	clSetKernelArg(simulationKernel, 5, sizeof(cl_float4), &lightSourcePosition);

	clEnqueueNDRangeKernel(queue, simulationKernel,
		1, NULL, &problemSize, &workGroupSize,
		0, NULL, NULL);
}

void visualizationStep() {
	cl_float16 clViewDir;
	float* camMatrix = camera.getViewDirMatrix().getPointer();
	for (int i = 0; i < 16; ++i) {
		clViewDir.s[i] = camMatrix[i];
	}

	Vector eye = camera.getEye();
	clViewDir.s[12] = eye.x;
	clViewDir.s[13] = eye.y;
	clViewDir.s[14] = eye.z;
	clViewDir.s[15] = 1.0f;

	clSetKernelArg(visualizationKernel, 0, sizeof(int), &windowWidth);
	clSetKernelArg(visualizationKernel, 1, sizeof(int), &windowHeight);
	clSetKernelArg(visualizationKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
	clSetKernelArg(visualizationKernel, 3, sizeof(int), &resolution);
	clSetKernelArg(visualizationKernel, 4, sizeof(cl_mem), &simulationBufferGPU);
	clSetKernelArg(visualizationKernel, 5, sizeof(int), &iteration);
	clSetKernelArg(visualizationKernel, 6, sizeof(cl_float16), &clViewDir);

	size_t visualizationBufferSize[2];
	visualizationBufferSize[0] = windowWidth;
	visualizationBufferSize[1] = windowHeight;

	clEnqueueNDRangeKernel(queue, visualizationKernel,
		2, NULL, visualizationBufferSize, NULL,
		0, NULL, NULL);

	clEnqueueReadBuffer(queue, visualizationBufferGPU, CL_TRUE, 0, sizeof(cl_float4) * windowWidth * windowHeight,
		visualizationBufferCPU, 0, NULL, NULL);

	glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// Iso surface raycasting
void isosurface() {
	cl_float16 clViewDir;
	float* camMatrix = camera.getViewDirMatrix().getPointer();
	for (int i = 0; i < 16; ++i) {
		clViewDir.s[i] = camMatrix[i];
	}

	Vector eye = camera.getEye();
	clViewDir.s[12] = eye.x;
	clViewDir.s[13] = eye.y;
	clViewDir.s[14] = eye.z;
	clViewDir.s[15] = 1.0f;

	clSetKernelArg(isosurfaceRaycastingKernel, 0, sizeof(int), &windowWidth);
	clSetKernelArg(isosurfaceRaycastingKernel, 1, sizeof(int), &windowHeight);
	clSetKernelArg(isosurfaceRaycastingKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
	clSetKernelArg(isosurfaceRaycastingKernel, 3, sizeof(int), &volumeSize[0]);
	clSetKernelArg(isosurfaceRaycastingKernel, 4, sizeof(cl_mem), &volumeDataGPU);
	clSetKernelArg(isosurfaceRaycastingKernel, 5, sizeof(float), &isoValue);
	clSetKernelArg(isosurfaceRaycastingKernel, 6, sizeof(cl_float16), &clViewDir);

	size_t visualizationBufferSize[2];
	visualizationBufferSize[0] = windowWidth;
	visualizationBufferSize[1] = windowHeight;

	clEnqueueNDRangeKernel(queue, isosurfaceRaycastingKernel,
		2, NULL, visualizationBufferSize, NULL,
		0, NULL, NULL);

	clEnqueueReadBuffer(queue, visualizationBufferGPU, CL_TRUE, 0, sizeof(cl_float4) * windowWidth * windowHeight,
		visualizationBufferCPU, 0, NULL, NULL);

	glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

// Alpha blended volume visualization
void alphaBlended() {
	cl_float16 clViewDir;
	float* camMatrix = camera.getViewDirMatrix().getPointer();
	for (int i = 0; i < 16; ++i) {
		clViewDir.s[i] = camMatrix[i];
	}

	Vector eye = camera.getEye();
	clViewDir.s[12] = eye.x;
	clViewDir.s[13] = eye.y;
	clViewDir.s[14] = eye.z;
	clViewDir.s[15] = 1.0f;

	clSetKernelArg(alphaBlendedKernel, 0, sizeof(int), &windowWidth);
	clSetKernelArg(alphaBlendedKernel, 1, sizeof(int), &windowHeight);
	clSetKernelArg(alphaBlendedKernel, 2, sizeof(cl_mem), &visualizationBufferGPU);
	clSetKernelArg(alphaBlendedKernel, 3, sizeof(int), &volumeSize[0]);
	clSetKernelArg(alphaBlendedKernel, 4, sizeof(cl_mem), &volumeDataGPU);
	clSetKernelArg(alphaBlendedKernel, 5, sizeof(float), &alphaExponent);
	clSetKernelArg(alphaBlendedKernel, 6, sizeof(float), &alphaCenter);
	clSetKernelArg(alphaBlendedKernel, 7, sizeof(cl_float16), &clViewDir);

	size_t visualizationBufferSize[2];
	visualizationBufferSize[0] = windowWidth;
	visualizationBufferSize[1] = windowHeight;

	clEnqueueNDRangeKernel(queue, alphaBlendedKernel,
		2, NULL, visualizationBufferSize, NULL,
		0, NULL, NULL);
		
	clEnqueueReadBuffer(queue, visualizationBufferGPU, CL_TRUE, 0, sizeof(cl_float4) * windowWidth * windowHeight,
		visualizationBufferCPU, 0, NULL, NULL);

	glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_FLOAT, visualizationBufferCPU);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	switch (method)
	{
	case 1:
		isosurface();
		break;
	case 2:
		alphaBlended();
		break;
	case 3:
		iteration++;
		simulationStep();
		visualizationStep();
		break;
	default:
		break;
	}
	glutSwapBuffers();
}

void animate()
{
	static float lastTime = 0.0f;
	long timeInMilliSecs = glutGet(GLUT_ELAPSED_TIME);
	float timeNow = timeInMilliSecs / 1000.0f;
	float deltaTime = timeNow - lastTime;

	camera.control(deltaTime, keysPressed);
	glutPostRedisplay();
}

void keyDown(unsigned char key, int x, int y)
{
	keysPressed[key] = true;
	switch (key)
	{
	case 'h':
		lightSourcePosition.s[0] -= 0.05f;
		iteration = 0;
		break;
	case 'k':
		lightSourcePosition.s[0] += 0.05f;
		iteration = 0;
		break;
	case 'u':
		lightSourcePosition.s[1] += 0.05f;
		iteration = 0;
		break;
	case 'j':
		lightSourcePosition.s[1] -= 0.05f;
		iteration = 0;
		break;
	case 'y':
		lightSourcePosition.s[2] += 0.05f;
		iteration = 0;
		break;
	case 'i':
		lightSourcePosition.s[2] -= 0.05f;
		iteration = 0;
		break;
	case 'r':
		iteration = 0;
		break;

	case '+':
		isoValue += 0.01f;
		break;
	case '-':
		isoValue -= 0.01f;
		break;

	case '[':
		alphaExponent *= 0.99f;
		break;
	case ']':
		alphaExponent *= 1.01f;
		break;

	case '{':
		alphaCenter -= 0.01f;
		break;
	case '}':
		alphaCenter += 0.01f;
		break;

	case '1':
		method = 1;
		break;
	case '2':
		method = 2;
		break;
	case '3':
		method = 3;
		break;
	}
}

void keyUp(unsigned char key, int x, int y)
{
	keysPressed[key] = false;
	switch (key)
	{
	case 27:
		exit(0);
		break;
	}
}

void mouseClick(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		camera.startDrag(x, y);
	}
}

void mouseMove(int x, int y)
{
	camera.drag(x, y);
}

void reshape(int newWidth, int newHeight)
{
	windowWidth = newWidth;
	windowHeight = newHeight;
	glViewport(0, 0, windowWidth, windowHeight);
	camera.setAspectRatio((float)windowWidth / windowHeight);
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(windowWidth, windowHeight);
	glutCreateWindow("Monte Carlo simulation");

	glutDisplayFunc(display);
	glutIdleFunc(animate);
	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyDown);
	glutKeyboardUpFunc(keyUp);
	glutMouseFunc(mouseClick);
	glutMotionFunc(mouseMove);

	glClearColor(0.17f, 0.4f, 0.6f, 1.0f);
	glDisable(GL_DEPTH_TEST);

	init();

	glutMainLoop();
	return 0;
}

