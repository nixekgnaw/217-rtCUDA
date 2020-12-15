// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  

#include <iostream>
#include <cuda_runtime.h>

// use CUDA's built-in float3 type��automated memory alignment & higher performance
#include "cutil_math.h" 
#include <vector_types.h>
#include "device_launch_parameters.h"

#define M_PI 3.14159265359f  // pi
#define WIDTH 512  // screenwidth
#define HEIGHT 384 // screenheight
#define SAMPS 200 // samples 

// __device__ : executed on the device (GPU) and callable only from the device


struct Ray {
    float3 o,d; // ���ߵ���ʼ�ͷ��� ray origin & direction 
    __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};

enum Refl_t 
{
    DIFF, 
    SPEC, 
    REFR };  // material types, used in radiance(), ��ǰֻ�������� ����֮��Ҫ��Ҫ�� only DIFF used here

struct Sphere {

    float rad;            // radius ��뾶
    float3 p, e, c; // ��Բ�� e? ��ɫposition, emission, colour 
    Refl_t refl;          // ���� reflection type (e.g. diffuse)
    //__device__ Sphere(float rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {} 
    __device__ float intersect(const Ray& r) const 
    { // returns distance, 0 if nohit
        float3 op = p - r.o;    //��һԪ���η��� ������ѧ Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        float t, eps = 0.0001f;  //����try 1e-4 epsilon required to prevent floating point precision artefacts
        float b = dot(op,r.d);    // b ��������
        float det = b * b - dot(op, op) + rad * rad;  // ������ѧdiscriminant quadratic equation
        if (det < 0) 
            return 0;       // ������ѧ
        else 
            det = sqrtf(det);    // sqrtf�� sqrt����н����ж�������
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0); // ȡclosest one
    }
};

// convert RGB float in range [0,1] to int in range [0, 255]
// perform gamma correction
inline float clamp(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
__device__ inline bool intersect_scene(const Ray& r, float& t, int& id)
{
    float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  // t is distance to closest intersection, initialise t to a huge number outside scene
    for (int i = int(n); i--;)  // test all scene objects for intersection
        if ((d = spheres[i].intersect(r)) && d < t)
        {  // if newly computed intersection distance d is smaller than current closest intersection distance
            t = d;  // keep track of distance along ray to closest intersection point 
            id = i; // and closest intersected object
        }
    return t < inf; // returns true if an intersection with the scene occurred, false when no hit
}

// !!����ֵֹ����������==
//random number generator from https://github.com/gz/rust-raytracer
__device__ static float getrandom(unsigned int* seed0, unsigned int* seed1) {
    *seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16);  // hash the seeds using bitwise AND and bitshifts
    *seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

    unsigned int ires = ((*seed0) << 16) + (*seed1);

    // Convert to float
    union {
        float f;
        unsigned int ui;
    } res;

    res.ui = (ires & 0x007fffff) | 0x40000000;  // bitwise AND, bitwise OR

    return (res.f - 2.f) / 2.f;
}

// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
__device__ float3 radiance(Ray& r, unsigned int* s1, unsigned int* s2) { // returns ray color

    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);

    // ray bounce loop (no Russian Roulette used) 
    for (int bounces = 0; bounces < 4; bounces++)
    {  // ��������ѭ�����ǵݹ�,����û�ö���˹���̴��ġ�����������Եݹ��Ч��(replaces recursion in CPU code)
        float t;           // distance to intersection
        int id = 0;        // id of intersected object
        if (!intersect_scene(r, t, id))
            return make_float3(0.0f, 0.0f, 0.0f); // û�򵽷��غ�ɫif miss, return black
        const Sphere& obj = spheres[id];  // the hit object
        float3 x = r.o + r.d * t;          // ���� hitpoint 
        float3 n = normalize(x - obj.p);    // ���� normal
        float3 nl = dot(n, r.d) < 0 ? n : n * -1; // ������Զ�����䷴����

        //����˹���̶ģ��Ĵ���
        //float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl
        //if (++depth > 5)
        //    if (curand_uniform(rand_state) < p)
        //        f = f * (1 / p);
        //    else
        //        return obj.e; //R.R.

        // add emission of current sphere to accumulated colour
        // (first term in rendering equation sum) 
        accucolor += mask * obj.e;
        if (obj.refl == DIFF)
        {

            float r1 = 2 * M_PI * getrandom(s1, s2); // ȡһ�������
            float r2 = getrandom(s1, s2);  // ȡ�ڶ��������
            float r2s = sqrtf(r2);
            //���ּ���
            float3 w = nl;
            float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
            float3 v = cross(w, u);
            float3 d = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrtf(1 - r2));

            // new ray origin is intersection point of previous ray with scene
            r.o = x + nl * 0.05f; // offset ray origin slightly to prevent self intersection
            r.d = d;
            mask *= obj.c;    // multiply with colour of object       
            mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
            mask *= 2;          // fudge factor
            //mask����ε��ڵݹ�Ŀ�������
        }
    }

    return accucolor;
}

// SCENE
// 9���� 9 spheres forming a Cornell box
// �����Ż�,����˵trick==�ó����ڴ���Ⱦ��small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = {
 { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
 { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
 { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
 { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
 { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
 { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
 { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 1
 { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, DIFF }, // small sphere 2
 { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
};

//�ؼ�����
__global__ void raytrac(float3* c) {

    // ÿ��thread��һ�����أ������е��˷ѣ�
    // replace CPU version for-loop    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (HEIGHT - y - 1) * WIDTH + x; // index of current pixel 

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;
    // first hardcoded camera ray(origin, direction) 
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); 
    float3 cx = make_float3(WIDTH * .5135 / HEIGHT, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.d)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r = make_float3(0.0f); // reset r to zero for every pixel  r is final pixel color       

    for (int s = 0; s < SAMPS; s++) {  // samples per pixel

     // compute primary ray direction
        float3 d = cx * ((.25 + x) / WIDTH - .5) + 
            cy * ((.25 + y) / HEIGHT - .5)+ cam.d;
        // create primary ray, add incoming radiance to pixelcolor
        r = r + radiance(Ray(cam.o + d * 40, normalize(d)), &s1, &s2) * (1. / SAMPS);
    }       // Camera rays are pushed ^^^^^ forward to start in interior 
    // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
    c[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {

    float3* c_h = new float3[WIDTH * HEIGHT]; // pointer to memory for image on the host (system RAM)
    //Sphere* spheres;
    float3* c_d;    // pointer to memory for image on the device (GPU VRAM)

    // allocate memory on the CUDA device (GPU VRAM)
    // �ó����ڴ�װ�����룬input�Ͳ����������ռ���
    gpuErrchk(cudaMalloc(&c_d, sizeof(float3) * WIDTH* HEIGHT));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(WIDTH / block.x, HEIGHT / block.y, 1);

    printf("CUDA initialised.\nStart rendering...\n");

    // schedule threads on device and launch CUDA kernel from host
    raytrac <<< grid, block >> > (c_d);
    gpuErrchk(cudaPeekAtLastError());

    // copy results of computation from device back to host
    cudaMemcpy(c_h, c_d, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(c_d);

    printf("Done!\n"); //���Ըĳɼ�ʱ

    // Write image to PPM file, a very simple image file format
    FILE* f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    for (int i = 0; i < WIDTH * HEIGHT; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", toInt(c_h[i].x),toInt(c_h[i].y),toInt(c_h[i].z));
    fclose(f);
    printf("Saved image to 'image.ppm'\n"); //���Ըĳɼ�ʱ

    delete[] c_h;

    return 0;
}