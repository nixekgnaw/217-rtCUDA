#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// use CUDA's built-in float3 type：automated memory alignment & higher performance
#include "cutil_math.h" 
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

#define M_PI 3.14159265359f  // pi
#define BLOCK_SIZE 8 // 2D block : BLOCK_SIZE*BLOCK_SIZE
struct Ray {
    float3 o,d; // 光线的起始和方向 ray origin & direction 
    __device__ Ray(float3 o_, float3 d_) : o(o_), d(d_) {}
};

enum Refl_t 
{
    DIFF, 
    SPEC, 
    REFR }; 

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
// ！！优化,或者说trick==用常量内存渲染球small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = {
 { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF }, //Left 
 { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF }, //Rght 
 { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Back 
 { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF }, //Frnt 
 { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Botm 
 { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF }, //Top 
 { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, SPEC }, // small sphere 1
 { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, REFR }, // small sphere 2
 { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f }, DIFF }  // Light
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

// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
__device__ float3 radiance(Ray& r, curandState* rs) { // returns ray color

    float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
    float3 mask = make_float3(1.0f, 1.0f, 1.0f);
    // ray bounce loop (no Russian Roulette used) 
    int bounces = 0;
    while (true)
    {  
        float t;           // distance to intersection
        int id = 0;        // id of intersected object
        if (!intersect_scene(r, t, id))
            return make_float3(0.0f, 0.0f, 0.0f); // 没打到返回黑色if miss, return black
        const Sphere& obj = spheres[id];  // the hit object
        float3 x = r.o + r.d * t;          // 交点 hitpoint 
        float3 n = normalize(x - obj.p);    // 法线 normal
        float3 nl = dot(n, r.d) < 0 ? n : n * -1; // 法线永远朝入射反方向
        float3 f = obj.c;

        //轮盘赌
        if (bounces++ > 5)
        {
            float rgbl = mask.x * 0.212671f + mask.y * 0.715160f + mask.z * 0.072169f;
            float p = min(0.95f, rgbl);
            if (curand_uniform(rs) >= p)
                break;
            mask /= p;
        }

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

            float r1 = 2 * M_PI * curand_uniform(rs); // 取一个随机数
            float r2 = curand_uniform(rs);  // 取第二个随机数
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
            //mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
            //mask *= 2;          // fudge factor

        }
        else if (obj.refl == SPEC)
        {
            r.o = x + nl * 0.05f;
            r.d = r.d - n * 2 * dot(n, r.d);
            mask *= obj.c;
        }
        else {
            bool into = dot(n,nl) > 0;                // Ray from outside going in?
            float nc = 1, nt = 1.5;
            float nnt = into ? nc / nt : nt / nc;
            float ddn = dot(r.d, nl);
            float cos2t = 1 - nnt * nnt * (1 - ddn * ddn);
            if (cos2t < 0) { /// Ideal dielectric REFRACTION
                r.o = x;
                r.d = r.d - n * 2 * dot(n, r.d);
                mask *= obj.c;
            }
            else {
                float3 tdir = normalize(r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t))));
                float a = nt - nc;
                float b = nt + nc;
                float R0 = a * a / (b * b);
                float c = 1 - (into ? -ddn : dot(tdir,n));

                float Re = R0 + (1 - R0) * c * c * c * c * c;
                float Tr = 1 - Re;
                float P = .25 + .5 * Re;
                float RP = Re / P;
                float TP = Tr / (1 - P);
                if ((bounces>2)&&(curand_uniform(rs) < P)) {
                    r.o = x;
                    r.d = r.d - n * 2 * dot(n, r.d);
                    mask *= obj.c * RP;
                }
                else {
                    r.o = x;
                    r.d = tdir;
                    mask *= obj.c * TP;
                }
            }
        }

    }

    return accucolor;
}

//关键代码
__global__ void raytrac(int samps,int height,int width,float3* c) {

    // 每个thread管一个像素
    // replace CPU version for-loop    
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int i = (height - y - 1) * width + x; // index of current pixel 

    curandState rs;
    curand_init(i, 0, 0, &rs);
    // first hardcoded camera ray(origin, direction) 
    Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); 
    float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
    float3 cy = normalize(cross(cx, cam.d)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
    float3 r = make_float3(0.0f); // reset r to zero for every pixel  r is final pixel color       

    for (int sy = 0;sy < 2;sy++) //for every pixel,
        for (int sx = 0; sx < 2; sx++)
        {
            for (int s = 0; s < samps; s++) {  // samples per pixel
                float r1 = 2 * curand_uniform(&rs);
                float dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                float r2 = 2 * curand_uniform(&rs);
                float dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);

                float3 d = cx * (((sx + .5 + dx) / 2 + x) / width - .5) +
                    cy * (((sy + .5 + dy) / 2 + y) / height - .5) + cam.d;
                Ray ray(cam.o + d * 40, normalize(d));
                r = r + radiance(ray, &rs) * (1. / samps);
            } // Camera rays are pushed ^^^^^ forward to start in interior
            c[i] = c[i] + make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f))*.25;

                

            // // compute primary ray direction
            //    float3 d = cx * ((.25 + x) / WIDTH - .5) +
            //        cy * ((.25 + y) / HEIGHT - .5) + cam.d;
            //    // create primary ray, add incoming radiance to pixelcolor
            //    Ray ray(cam.o + d * 40, normalize(d));
            //    r = r + radiance(ray, &rs) * (1. / samps);
            //}       // Camera rays are pushed ^^^^^ forward to start in interior 
            //// write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
            //c[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
            
        }
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

int main(int argc, char* argv[]) {
    int samps = argc >= 2 ? atoi(argv[1])/4 : 50;
    int width = argc == 4 ? atoi(argv[2]): 1024; // screenwidth
    int height= argc == 4 ? atoi(argv[3]): 768; // screenheight
    float3* c_h = new float3[width * height]; // pointer to memory for image on the host (system RAM)
    //Sphere* spheres;
    float3* c_d;    // pointer to memory for image on the device (GPU VRAM)

    // allocate memory on the CUDA device (GPU VRAM)
    // 用常量内存装了输入，input就不用另外分配空间了
    gpuErrchk(cudaMalloc(&c_d, sizeof(float3) * width* height));

    // dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
    dim3 dim_grid, dim_block;
    //dimblock
    dim_block.x = BLOCK_SIZE;
    dim_block.y = BLOCK_SIZE;
    dim_block.z = 1;
    //dimgrid
    dim_grid.x = (width - 1) / BLOCK_SIZE + 1;
    dim_grid.y = (height - 1) / BLOCK_SIZE + 1;
    dim_grid.z = 1;

    printf("CUDA initialised.\nStart rendering...\n");

    // schedule threads on device and launch CUDA kernel from host
    raytrac <<< dim_grid, dim_block >>> (samps,height,width,c_d);
    gpuErrchk(cudaPeekAtLastError());

    // copy results of computation from device back to host
    cudaMemcpy(c_h, c_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

    // free CUDA memory
    cudaFree(c_d);

    printf("Done!\n"); //可以改成计时

    // Write image to PPM file, a very simple image file format
    FILE* f = fopen("image.ppm", "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    for (int i = 0; i < width * height; i++)  // loop over pixels, write RGB values
        fprintf(f, "%d %d %d ", toInt(c_h[i].x),toInt(c_h[i].y),toInt(c_h[i].z));
    fclose(f);
    printf("Saved image to 'image.ppm'\n"); //可以改成计时

    delete[] c_h;

    return 0;
}
