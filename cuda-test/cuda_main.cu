// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  

#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h

#define M_PI 3.14159265359f  // pi
#define width 512  // screenwidth
#define height 384 // screenheight
#define samps 200 // samples 

// __device__ : executed on the device (GPU) and callable only from the device


struct Ray {
    float3 o,d; // 光线的起始和方向 ray origin & direction 
    __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};

enum Refl_t 
{
    DIFF, 
    SPEC, 
    REFR };  // material types, used in radiance(), 当前只有漫反射 看看之后要不要加 only DIFF used here

struct Sphere {

    float rad;            // radius 球半径
    float3 p, e, c; // 球圆心 e? 颜色position, emission, colour 
    Refl_t refl;          // 材质 reflection type (e.g. diffuse)
    //__device__ Sphere(float rad_, float3 p_, float3 e_, float3 c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) {} 
    __device__ float intersect(const Ray& r) const 
    { // returns distance, 0 if nohit
        float3 op = p - r.o;    //解一元二次方程 初中数学 Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        float t, eps = 0.0001f;  //可以try 1e-4 epsilon required to prevent floating point precision artefacts
        float b = dot(op,r.d);    // b 方程所需
        float det = b * b - dot(op, op) + rad * rad;  // 初中数学discriminant quadratic equation
        if (det < 0) 
            return 0;       // 初中数学
        else 
            det = sqrtf(det);    // sqrtf和 sqrt差别？有解则判断正负根
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0); // 取closest one
    }
};

// SCENE
// 9个球 9 spheres forming a Cornell box
// ！！优化,或者说作弊==用常量内存渲染球small enough to be in constant GPU memory
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

//__device__ inline float clamp(float x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
//__device__ inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
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

// !!奇奇怪怪的随机数生成==
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
    {  // ！！用了循坏不是递归,而且没用俄罗斯轮盘待改。可以另外测试递归的效果(replaces recursion in CPU code)
        float t;           // distance to intersection
        int id = 0;        // id of intersected object
        if (!intersect_scene(r, t, id))
            return make_float3(0.0f, 0.0f, 0.0f); // 没打到返回黑色if miss, return black
        const Sphere& obj = spheres[id];  // the hit object
        float3 x = r.o + r.d * t;          // 交点 hitpoint 
        float3 n = normalize(x - obj.p);    // 法线 normal
        float3 nl = dot(n, r.d) < 0 ? n : n * -1; // 法线永远朝入射反方向

        //俄罗斯轮盘赌？的代码
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
      
            float r1 = 2 * M_PI * getrandom(s1, s2); // 取一个随机数
            float r2 = getrandom(s1, s2);  // 取第二个随机数
            float r2s = sqrtf(r2);
            //各种计算
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
        }
    }

    return accucolor;
}


// __global__ : executed on the device (GPU) and callable only from host (CPU) 
// this kernel runs in parallel on all the CUDA threads

__global__ void render_kernel(float3* output) {

    // assign a CUDA thread to every pixel (x,y) 
    // blockIdx, blockDim and threadIdx are CUDA specific keywords
    // replaces nested outer loops in CPU code looping over image rows and image columns 
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1) * width + x; // index of current pixel (calculated using thread index) 

    unsigned int s1 = x;  // seeds for random number generator
    unsigned int s2 = y;

    // generate ray directed at lower left corner of the screen
    // compute directions for all other rays by adding cx and cy increments in x and y direction
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction) 
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.d)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r; // r is final pixel color       

    r = make_float3(0.0f); // reset r to zero for every pixel 

    for (int s = 0; s < samps; s++) {  // samples per pixel

     // compute primary ray direction
        float3 d = cam.d + cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5);

        // create primary ray, add incoming radiance to pixelcolor
        r = r + radiance(Ray(cam.o + d * 40, normalize(d)), &s1, &s2) * (1. / samps);
    }       // Camera rays are pushed ^^^^^ forward to start in interior 

    // write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
    output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction

int main() {

    float3* output_h = new float3[width * height]; // pointer to memory for image on the host (system RAM)
    float3* output_d;    // pointer to memory for image on the device (GPU VRAM)

    // allocate memory on the CUDA device (GPU VRAM)
    cudaMalloc(&output_d, width * height * sizeof(float3));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 block(8, 8, 1);
    dim3 grid(width / block.x, height / block.y, 1);

    printf("CUDA initialised.\nStart rendering...\n");

    // schedule threads on device and launch CUDA kernel from host
    render_kernel << < grid, block >> > (output_d);

    // copy results of computation from device back to host
    cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(output_d);

    printf("Done!\n");

    // Write image to PPM file, a very simple image file format
    FILE* f = fopen("smallptcuda.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", toInt(output_h[i].x),
            toInt(output_h[i].y),
            toInt(output_h[i].z));

    printf("Saved image to 'smallptcuda.ppm'\n");

    delete[] output_h;
    system("PAUSE");
}