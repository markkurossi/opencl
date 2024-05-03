/*
 * simple.c
 */

#include <stdio.h>
#include <OpenCL/opencl.h>

#include <mach/mach_time.h>
#include <stdint.h>

#define DATA_SIZE (1024)

const char *KernelSource =
  "__kernel void square(__global float* input, __global float* output, const unsigned int count) { \n" \
  "   int i = get_global_id(0);                                                                    \n" \
  "   if(i < count) { output[i] = input[i] * input[i]; }                                           \n" \
  "}";

int main(void)
{
  int err;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel kernel;
  cl_mem input;
  cl_mem output;
  float data[DATA_SIZE];
  unsigned int count = DATA_SIZE;
  size_t local;
  size_t global;
  float results[DATA_SIZE];
  unsigned int correct = 0;
  uint64_t ts;

  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  commands = clCreateCommandQueue(context, device_id, 0, &err);
  program = clCreateProgramWithSource(context, 1,
                                      (const char **) & KernelSource, NULL,
                                      &err);
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  kernel = clCreateKernel(program, "square", &err);
  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE,
                         NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE,
                          NULL, NULL);


  for (int i = 0; i < DATA_SIZE; i++)
    data[i] = i;

  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0,
                             sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);

  clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
  clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(local), &local, NULL);
  global = count;

  ts = mach_absolute_time();

  clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0,
                         NULL, NULL);
  clFinish(commands);
  ts = mach_absolute_time() - ts;
  printf("OpenCL: %lld\n", ts);

  clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count,
                      results, 0, NULL, NULL);


  ts = mach_absolute_time();

  for (int i = 0; i < count; i++)
      if (results[i] == data[i] * data[i])
        correct++;

  ts = mach_absolute_time() - ts;
  printf("CPU: %lld\n", ts);

  printf("Computed '%d/%d' correct values!\n", correct, count);

  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  return 0;
}
