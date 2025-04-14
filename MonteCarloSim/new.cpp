// MonteCarlo.cpp : Defines the entry point for the console application.

#include <string>
#include <vector>
#include <iostream>
#include "Common.h"

// OpenCL C API
#include <CL/opencl.h>

// OpenCL C++ API
#include "cl.hpp"

// GLUT
#include <GL/glut.h>  // for linux

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

const int numPhotons = 10000;  // Define the number of photons for the simulation
const int GRID_WIDTH = 128;    // Define grid dimensions (adjust as needed)
const int GRID_HEIGHT = 128;
const int GRID_DEPTH = 128;

cl::Device device = devices[0];  // devices is an array of OpenCL devices

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

void init() {
    // Minimal OpenCL infrastructure
    clGetPlatformIDs(1, &platformID, NULL);
    clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, NULL);
    cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0 };
    context = clCreateContext(contextProperties, 1, &deviceID, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Visualization buffers
    visualizationBufferCPU = new cl_float4[windowWidth * windowHeight];
    visualizationBufferGPU = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_float4) * windowWidth * windowHeight, NULL, NULL);

    // IsoSurface raycasting
    std::string source = FileToString("../kernels/visualization.cl");
    const char* csource = source.c_str();

    visualizationProgram = clCreateProgramWithSource(context, 1, &csource, NULL, NULL);
    cl_int err = clBuildProgram(visualizationProgram, 1, &deviceID, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
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
    if (err != CL_SUCCESS) {
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
    for (int i = 0; i < workGroupSize * maxComputeUnits; ++i) {
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

void resetSimulation() {
    std::vector<float> zeroPhotonData(numPhotons * 4, 0.0f); // x, y, z, energy
    std::vector<float> zeroEnergyData(GRID_WIDTH * GRID_HEIGHT * GRID_DEPTH, 0.0f);

    cl::CommandQueue queue(context, device);

    // Clear photon buffer
    queue.enqueueWriteBuffer(photonsBuffer, CL_TRUE, 0, sizeof(float) * zeroPhotonData.size(), zeroPhotonData.data());

    // Clear energy grid
    queue.enqueueWriteBuffer(energyBuffer, CL_TRUE, 0, sizeof(float) * zeroEnergyData.size(), zeroEnergyData.data());

    // Reset RNG seeds
queue.enqueueWriteBuffer(seedBuffer, CL_TRUE, 0, sizeof(uint) * zeroEnergyData.size(), zeroEnergyData);
}
int main(int argc, char** argv) { init(); std::cout << "OpenCL initialization complete. Ready for simulation!" << std::endl; glutMainLoop(); return 0; }