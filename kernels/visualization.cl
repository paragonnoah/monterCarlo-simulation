// -*- mode: c++ -*-
//**************
// Visualization
//**************

struct ray {
  float4 origin;
  float4 direction;
};

// intersect ray with a box
int intersectBox(float4 r_o, float4 r_d, float4 boxmin, float4 boxmax, float *tnear, float *tfar)
{
  float4 invR = (float4)(1.0f) / r_d;
  float4 tbot = invR * (boxmin - r_o);
  float4 ttop = invR * (boxmax - r_o);

  float4 tmin = min(ttop, tbot);
  float4 tmax = max(ttop, tbot);

  float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
  float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));

  *tnear = largest_tmin;
  *tfar = smallest_tmax;

  return smallest_tmax > largest_tmin;
}

float getDensityFromVolume(const float4 p, const int resolution, __global float* volumeData)
{
  int x = p.x * resolution;
  int y = p.y * resolution;
  int z = p.z * resolution;

  if (x < 0 || x >= resolution) return 0.0f;
  if (y < 0 || y >= resolution) return 0.0f;
  if (z < 0 || z >= resolution) return 0.0f;

  return volumeData[x + y * resolution + z * resolution * resolution];
}

float4 getNormalFromVolume(const float4 p, const int resolution, __global float* volumeData)
{
  float4 normal;

  normal.x = getDensityFromVolume((float4)(p.x + 2.0f / resolution, p.y, p.z, 0.0f), resolution, volumeData)
           - getDensityFromVolume((float4)(p.x - 2.0f / resolution, p.y, p.z, 0.0f), resolution, volumeData);
  normal.y = getDensityFromVolume((float4)(p.x, p.y + 2.0f / resolution, p.z, 0.0f), resolution, volumeData)
           - getDensityFromVolume((float4)(p.x, p.y - 2.0f / resolution, p.z, 0.0f), resolution, volumeData);
  normal.z = getDensityFromVolume((float4)(p.x, p.y, p.z + 2.0f / resolution, 0.0f), resolution, volumeData)
           - getDensityFromVolume((float4)(p.x, p.y, p.z - 2.0f / resolution, 0.0f), resolution, volumeData);
  normal.w = 0.0f;

  if (dot(normal, normal) < 0.001f)
    normal = (float4)(0.0f, 0.0f, 1.0f, 0.0f);

  return normalize(normal);
}
__kernel void isosurfaceWithShading(const int width, const int height, __global float4* visualizationBuffer,
                                     const int resolution, __global float* volumeData,
                                     const float isoValue, const float16 invViewMatrix)
{
  int2 id = (int2)(get_global_id(0), get_global_id(1));
  float2 uv = (float2)((id.x / (float)width) * 2.0f - 1.0f, (id.y / (float)height) * 2.0f - 1.0f);

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f, 1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

  struct ray eyeRay;
  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize((float4)(uv.x, uv.y, -2.0f, 0.0f));
  eyeRay.direction.x = dot(temp, (float4)(invViewMatrix.s0, invViewMatrix.s1, invViewMatrix.s2, invViewMatrix.s3));
  eyeRay.direction.y = dot(temp, (float4)(invViewMatrix.s4, invViewMatrix.s5, invViewMatrix.s6, invViewMatrix.s7));
  eyeRay.direction.z = dot(temp, (float4)(invViewMatrix.s8, invViewMatrix.s9, invViewMatrix.sA, invViewMatrix.sB));
  eyeRay.direction.w = 0.0f;

  // Define light direction and normalize it
  float4 lightDir = normalize((float4)(0.3f, -2.0f, 0.0f, 0.0f));

  float4 color = (float4)(0.0f);
  float tnear, tfar;

  if (intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar))
  {
    float stepSize = 0.01f;
    float t = tnear;
    float4 currentPoint = eyeRay.origin + eyeRay.direction * t;

    while (t < tfar)
    {
      float density = getDensityFromVolume(currentPoint, resolution, volumeData);
      if (density > isoValue)
      {
        float4 normal = getNormalFromVolume(currentPoint, resolution, volumeData);

        // Compute the diffuse shading using the dot product between normal and light direction
        float diffuse = clamp(dot(normalize(normal), lightDir), 0.0f, 1.0f);

        // The color is white with the computed diffuse value
        color = (float4)(diffuse, diffuse, diffuse, 1.0f);
        break;
      }
      currentPoint += eyeRay.direction * stepSize;
      t += stepSize;
    }
  }

  if (id.x < width && id.y < height)
    visualizationBuffer[id.x + id.y * width] = color;
}

__kernel void isosurface(const int width, const int height, __global float4* visualizationBuffer,
                         const int resolution, __global float* volumeData,
                         const float isoValue, const float16 invViewMatrix)
{
  int2 id = (int2)(get_global_id(0), get_global_id(1));
  float2 uv = (float2)((id.x / (float)width) * 2.0f - 1.0f, (id.y / (float)height) * 2.0f - 1.0f);

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f, 1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

  struct ray eyeRay;
  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize((float4)(uv.x, uv.y, -2.0f, 0.0f));
  eyeRay.direction.x = dot(temp, (float4)(invViewMatrix.s0, invViewMatrix.s1, invViewMatrix.s2, invViewMatrix.s3));
  eyeRay.direction.y = dot(temp, (float4)(invViewMatrix.s4, invViewMatrix.s5, invViewMatrix.s6, invViewMatrix.s7));
  eyeRay.direction.z = dot(temp, (float4)(invViewMatrix.s8, invViewMatrix.s9, invViewMatrix.sA, invViewMatrix.sB));
  eyeRay.direction.w = 0.0f;

  float4 color = (float4)(0.0f);
  float tnear, tfar;

  if (intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar))
  {
    float stepSize = 0.01f;
    float t = tnear;
    float4 currentPoint = eyeRay.origin + eyeRay.direction * t;

    while (t < tfar)
    {
      float density = getDensityFromVolume(currentPoint, resolution, volumeData);
      if (density > isoValue)
   if (density > isoValue)
{
  color = (float4)(1.0f, 1.0f, 1.0f, 1.0f); // constant white
  break;
}

      currentPoint += eyeRay.direction * stepSize;
      t += stepSize;
    }
  }

  if (id.x < width && id.y < height)
    visualizationBuffer[id.x + id.y * width] = color;
}

__kernel void alphaBlended(const int width, const int height, __global float4* visualizationBuffer,
                           const int resolution, __global float* volumeData,
                           const float alphaExponent, const float alphaCenter,
                           const float16 invViewMatrix)
{
  int2 id = (int2)(get_global_id(0), get_global_id(1));
  float2 uv = (float2)((id.x / (float)width) * 2.0f - 1.0f, (id.y / (float)height) * 2.0f - 1.0f);

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f, 1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

  struct ray eyeRay;
  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize((float4)(uv.x, uv.y, -2.0f, 0.0f));
  eyeRay.direction.x = dot(temp, (float4)(invViewMatrix.s0, invViewMatrix.s1, invViewMatrix.s2, invViewMatrix.s3));
  eyeRay.direction.y = dot(temp, (float4)(invViewMatrix.s4, invViewMatrix.s5, invViewMatrix.s6, invViewMatrix.s7));
  eyeRay.direction.z = dot(temp, (float4)(invViewMatrix.s8, invViewMatrix.s9, invViewMatrix.sA, invViewMatrix.sB));
  eyeRay.direction.w = 0.0f;

  float4 sum = (float4)(0.0f);
  float tnear, tfar;

  if (intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar))
  {
    float stepSize = 0.01f;
    float t = tnear;
    float4 currentPoint = eyeRay.origin + eyeRay.direction * t;

    while (t < tfar)
    {
      float density = getDensityFromVolume(currentPoint, resolution, volumeData);
      float alpha = pow(fabs(density - alphaCenter), alphaExponent) * stepSize;

      float4 sampleColor = (float4)(density, density, density, 1.0f);
      sum = (1.0f - alpha) * sum + alpha * sampleColor;

      currentPoint += eyeRay.direction * stepSize;
      t += stepSize;
    }
  }

  if (id.x < width && id.y < height)
    visualizationBuffer[id.x + id.y * width] = sum;
}

__kernel void xrayRender(const int width, const int height,
                         __global float4* visualizationBuffer,
                         const int resolution, __global float* volumeData,
                         const float alphaExponent, const float16 invViewMatrix)
{
  int2 id = (int2)(get_global_id(0), get_global_id(1));
  float2 uv = (float2)((id.x / (float)width) * 2.0f - 1.0f, (id.y / (float)height) * 2.0f - 1.0f);

  float4 boxMin = (float4)(-1.0f, -1.0f, -1.0f, 1.0f);
  float4 boxMax = (float4)(1.0f, 1.0f, 1.0f, 1.0f);

  // Setup ray
  struct ray eyeRay;
  eyeRay.origin = (float4)(invViewMatrix.sC, invViewMatrix.sD, invViewMatrix.sE, invViewMatrix.sF);

  float4 temp = normalize((float4)(uv.x, uv.y, -2.0f, 0.0f));
  eyeRay.direction.x = dot(temp, (float4)(invViewMatrix.s0, invViewMatrix.s1, invViewMatrix.s2, invViewMatrix.s3));
  eyeRay.direction.y = dot(temp, (float4)(invViewMatrix.s4, invViewMatrix.s5, invViewMatrix.s6, invViewMatrix.s7));
  eyeRay.direction.z = dot(temp, (float4)(invViewMatrix.s8, invViewMatrix.s9, invViewMatrix.sA, invViewMatrix.sB));
  eyeRay.direction.w = 0.0f;

  float tnear, tfar;
  float4 color = (float4)(0.0f);

  if (intersectBox(eyeRay.origin, eyeRay.direction, boxMin, boxMax, &tnear, &tfar))
  {
    float stepSize = 0.01f;
    float t = tnear;
    float4 currentPoint = eyeRay.origin + eyeRay.direction * t;

    while (t < tfar)
    {
      float density = getDensityFromVolume(currentPoint, resolution, volumeData);
      float alpha = pow(density, alphaExponent) * stepSize;

      // White X-ray accumulation
      color = (1.0f - alpha) * color + alpha * (float4)(1.0f, 1.0f, 1.0f, 1.0f);

      currentPoint += eyeRay.direction * stepSize;
      t += stepSize;

      // Early termination if color is saturated
      if (color.x > 0.99f && color.y > 0.99f && color.z > 0.99f)
        break;
    }

    color.w = 1.0f; // set alpha to 1 for output buffer
  }

  if (id.x < width && id.y < height)
    visualizationBuffer[id.x + id.y * width] = color;
}
