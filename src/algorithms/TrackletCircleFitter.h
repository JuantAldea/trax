#pragma once

#include <tuple>

#include <clever/clever.hpp>
#include <datastructures/TrackletCollection.h>
#include <datastructures/KernelWrapper.hpp>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

class TrackletCircleFitter : public KernelWrapper<TrackletCircleFitter>
{
public:
    TrackletCircleFitter(clever::context & context) :
        KernelWrapper(context),
        trackletCircleFitterStore(context)
    {
        PLOG << "trackletCircleFitterStore WorkGroupSize: "
             << trackletCircleFitterStore.getWorkGroupSize() << std::endl;
    }

    clever::vector<float, 1>* run(const HitCollection &hits,
                                  const TrackletCollection &tracklets,
                                  const clever::vector<uint, 1> &validTrackletsIndices,
                                  uint nThreads,
                                  bool printPROLIX = false) const;


    KERNEL_CLASS(trackletCircleFitterStore,
        __kernel void trackletCircleFitterStore(
            //input
            __global float * const __restrict hitX,
            __global float * const __restrict hitY,
            __global float * const __restrict hitZ,
            __global uint * const __restrict h0,
            __global uint * const __restrict h1,
            __global uint * const __restrict h2,
            __global uint * const __restrict validTrackletsIndices,
            //output
            __global float * const __restrict tripletsPt,
            //workload
            const uint nValidTriplets)
    {
        const size_t gid = get_global_id(0);

        if (gid >= nValidTriplets) {
            return;
        }

        const uint tripletIndex = validTrackletsIndices[gid];
        const uint innerHit = h0[tripletIndex];
        const uint middleHit = h1[tripletIndex];
        const uint outerHit = h2[tripletIndex];
        // TODO this code needs to be benchmarked. opencl float3 are actually float4
        // so it might be wasting registers.
        const float3 hit0 = (float3)(hitX[innerHit], hitY[innerHit], hitZ[innerHit]);
        const float3 hit1 = (float3)(hitX[middleHit], hitY[middleHit], hitZ[middleHit]);
        const float3 hit2 = (float3)(hitX[outerHit], hitY[outerHit], hitZ[outerHit]);

        //calculate Pt
        const float3 pP0 = (float3)(hit0.x, hit0.y, hit0.x * hit0.x + hit0.y * hit0.y);
        const float3 pP1 = (float3)(hit1.x, hit1.y, hit1.x * hit1.x + hit1.y * hit1.y);
        const float3 pP2 = (float3)(hit2.x, hit2.y, hit2.x * hit2.x + hit2.y * hit2.y);
        //const float3 a(pP1.x - pP0.x, pP1.y - pP0.y, pP1.z - pP0.z);
        //const float3 b(pP2.x - pP0.x, pP2.y - pP0.y, pP2.z - pP0.z);
        //const float3 n(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
        const float3 a = (float3)pP1 - pP0;
        const float3 b = (float3)pP2 - pP0;
        const float3 n = cross(a, b);
        //const float n_module = sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
        //const float3 unit_n(n.x / n_module, n.y / n_module, n.z / n_module);
        const float3 unit_n = normalize(n);
        //TIP
        //const float2 circle_center = (float2)(-unit_n.x / (2 * unit_n.z), -unit_n.y / (2 * unit_n.z));

        const float c = -dot(unit_n, pP0);
        //const float c = -(unit_n.x * pP0.x + unit_n.y * pP0.y + unit_n.z * pP0.z);
        const float unit_n_z_squared = unit_n.z * unit_n.z;
        const float circle_radius = sqrt((1 - unit_n_z_squared - 4 * c * unit_n.z) / (4 * unit_n_z_squared));
        // Ideal Magnetic Field [T] (-0,-0, 3.8112)
        const float BZ = 3.8112;
        // e = 1.602177×10^-19 C (coulombs)
        const float Q = 1.602177E-19;
        // c = 2.998×10^8 m/s (meters per second)
        //const float C = 2.998E8;
        // 1 GeV/c = 5.344286×10^-19 J s/m (joule seconds per meter)
        const float GEV_C = 5.344286E-19;

        // as we are comparing p_t difference, we don't care about linear transformations
        // so we could skip calculating the products and divisions and stick to the bare radius
        // even since we are just comparing for mininum pt difference we could skip the sqrt
        // as well since sqrt is a monotonic increasing function.
        //tripletsPt[tripletIndex] = Q * BZ * (circle_radius * 1E-2) / GEV_C;
        tripletsPt[tripletIndex] = circle_radius;
    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
    cl_mem,
    cl_uint);
};
