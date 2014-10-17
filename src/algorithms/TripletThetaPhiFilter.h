#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>

#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Pairings.h>
#include <datastructures/TripletConfiguration.h>
#include <datastructures/Grid.h>
#include <datastructures/Logger.h>

#include <datastructures/KernelWrapper.hpp>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

/*
    Class which implements to most dump tracklet production possible - combine
    every hit with every other hit.
    The man intention is to test the data transfer and the data structure
 */
class TripletThetaPhiFilter: public KernelWrapper<TripletThetaPhiFilter>
{

public:

    TripletThetaPhiFilter(clever::context & ctext) :
        KernelWrapper(ctext),
        filterCount(ctext),
        filterPopCount(ctext),
        filterStore(ctext),
        filterOffsetStore(ctext),
        filterOffsetMonotonizeStore(ctext)
    {
        // create the buffers this algorithm will need to run
        PLOG << "FilterKernel WorkGroupSize: " << filterCount.getWorkGroupSize() << std::endl;
        PLOG << "StoreKernel WorkGroupSize: " << filterStore.getWorkGroupSize() << std::endl;
    }

    TrackletCollection * run(HitCollection & hits, const Grid & grid,
                             const Pairing & pairs, const Pairing & tripletCandidates,
                             int nThreads, const TripletConfigurations & layerTriplets, bool printProlix = false);

    KERNEL_CLASSP(filterCount,
                  oclDEFINES,
                  __kernel void filterCount(
                      //configuration
                      __global const float * thetaCut, __global const float * phiCut, __global const float * maxTIP,
                      const float minRadius,
                      // hit input
                      __global const uint2 * pairs,
                      __global const uint2 * triplets, const uint nTriplets,
                      __global const float * hitGlobalX, __global const float * hitGlobalY,
                      __global const float * hitGlobalZ,
                      __global const uint * hitEvent, __global const uint * hitLayer,
                      // intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
                      __global uint * oracle)
    {
        size_t gid = get_global_id(0); // thread

        if (gid < nTriplets) {

            uint firstHit = pairs[triplets[gid].x].x;
            uint secondHit = pairs[triplets[gid].x].y;
            uint thirdHit = triplets[gid].y;
            // must be the same as hitEvent[secondHit] --> ensured during pair building
            uint event = hitEvent[firstHit];
            //the layerTriplet is defined by its innermost layer
            uint layerTriplet = hitLayer[firstHit] - 1;

            float dThetaCut = thetaCut[layerTriplet];
            float dPhiCut = phiCut[layerTriplet];
            float tipCut = maxTIP[layerTriplet];

            bool valid = true;

            //tanTheta1
            float angle1 = atan2(sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit]) *
                                      (hitGlobalX[secondHit] - hitGlobalX[firstHit])
                                      + (hitGlobalY[secondHit] - hitGlobalY[firstHit]) * (hitGlobalY[secondHit] -
                                              hitGlobalY[firstHit]))
                                 , (hitGlobalZ[secondHit] - hitGlobalZ[firstHit]));
            //tanTheta2
            float angle2 = atan2(sqrt((hitGlobalX[thirdHit] - hitGlobalX[secondHit]) *
                                      (hitGlobalX[thirdHit] - hitGlobalX[secondHit])
                                      + (hitGlobalY[thirdHit] - hitGlobalY[secondHit]) * (hitGlobalY[thirdHit] -
                                              hitGlobalY[secondHit]))
                                 , (hitGlobalZ[thirdHit] - hitGlobalZ[secondHit]));
            float delta = fabs(angle2 / angle1);
            valid = valid * (1 - dThetaCut <= delta && delta <= 1 + dThetaCut);

            //tanPhi1
            angle1 = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) ,
                           (hitGlobalX[secondHit] - hitGlobalX[firstHit]));
            //tanPhi2
            angle2 = atan2((hitGlobalY[thirdHit] - hitGlobalY[secondHit]) ,
                           (hitGlobalX[thirdHit] - hitGlobalX[secondHit]));

            delta = angle2 - angle1;
            delta += (delta > M_PI_F) ? -2 * M_PI_F : (delta < -M_PI_F) ? 2 * M_PI_F : 0; //fix wrap around
            valid = valid * (fabs(delta) <= dPhiCut);

            //circle fit
            //map points to parabloid: (x,y) -> (x,y,x^2+y^2)
            float3 pP1 = (float3)(hitGlobalX[firstHit],
                                  hitGlobalY[firstHit],
                                  hitGlobalX[firstHit] * hitGlobalX[firstHit] + hitGlobalY[firstHit] * hitGlobalY[firstHit]);

            float3 pP2 = (float3)(hitGlobalX[secondHit],
                                  hitGlobalY[secondHit],
                                  hitGlobalX[secondHit] * hitGlobalX[secondHit] + hitGlobalY[secondHit] * hitGlobalY[secondHit]);

            float3 pP3 = (float3)(hitGlobalX[thirdHit],
                                  hitGlobalY[thirdHit],
                                  hitGlobalX[thirdHit] * hitGlobalX[thirdHit] + hitGlobalY[thirdHit] * hitGlobalY[thirdHit]);

            //span two vectors
            float3 a = pP2 - pP1;
            float3 b = pP3 - pP1;

            //compute unit cross product
            float3 n = cross(a, b);
            n = normalize(n);

            //formula for orign and radius of circle from Strandlie et al.
            float2 cOrigin = (float2)((-n.x) / (2 * n.z),
                                      (-n.y) / (2 * n.z));

            float c = -(n.x * pP1.x + n.y * pP1.y + n.z * pP1.z);

            float cR = sqrt((1 - n.z * n.z - 4 * c * n.z) / (4 * n.z * n.z));

            //find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
            float2 v = -cOrigin;
            v = normalize(v);
            float2 pCA = (float2)(cOrigin.x + cR * v.x,
                                  cOrigin.y + cR * v.y);

            //TIP = distance of point of closest approach to origin
            float tip_squared = pCA.x * pCA.x + pCA.y * pCA.y;

            valid = valid * (tip_squared <= (tipCut * tipCut));

            //check for compliance with minimum radius, ie. min Pt

            valid = valid * (cR >= minRadius);

            atomic_or(&oracle[(gid) / 32], (valid << (gid % 32)));
        }
    },
    cl_mem, cl_mem, cl_mem, cl_float,
    cl_mem,
    cl_mem,  cl_uint,
    cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_mem);

    KERNEL_CLASSP(filterPopCount, oclDEFINES,

                  __kernel void filterPopCount(__global const uint * oracle, __global uint * prefixSum,
                          const uint n)
    {

        size_t gid = get_global_id(0);
        if (gid < n) {
            // only present in OCL 1.2 prefixSum[gid] = popcount(oracle[gid]);
            uint i = oracle[gid];
            i = i - ((i >> 1) & 0x55555555);
            i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
            prefixSum[gid] = (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
        }

    }, cl_mem, cl_mem, cl_uint);

    KERNEL_CLASSP(filterStore,
                  oclDEFINES,
                  __kernel void filterStore(
                      // hit input
                      __global const uint2 * pairs, const uint nLayerTriplets,
                      __global const uint2 * triplets,
                      // input for oracle and prefix sum
                      __global const uint * oracle, const uint nOracleBytes, __global const uint * prefixSum,
                      // output of tracklet data
                      __global uint * trackletHitId1, __global uint * trackletHitId2, __global uint * trackletHitId3,
                      __global uint * trackletOffsets)
    {
        size_t gid = get_global_id(0); // thread

        if (gid < nOracleBytes) {
            uint pos = prefixSum[gid]; //first position to write
            uint nextThread = prefixSum[gid + 1]; //first position of next thread
            //configure oracle
            uint lOracle = oracle[gid]; //load oracle byte
            //loop over bits of oracle byte
            for (uint i = 0; i < 32 /*&& pos < nextThread*/;
                    ++i) { // pos < prefixSum[id+1] can lead to thread divergence
                //is this a valid triplet?
                bool valid = lOracle & (1 << i);
                //last triplet written on [pos] is valid one
                if (valid) {
                    trackletHitId1[pos] = pairs[triplets[gid * 32 + i].x].x;
                    trackletHitId2[pos] = pairs[triplets[gid * 32 + i].x].y;
                    trackletHitId3[pos] = triplets[gid * 32 + i].y;
                }
                //advance pos if valid
                pos += valid;
            }
        }
    }, cl_mem, cl_uint,
    cl_mem,
    cl_mem, cl_uint, cl_mem,
    cl_mem, cl_mem, cl_mem,
    cl_mem);

    KERNEL_CLASSP(filterOffsetStore,
                  oclDEFINES,

                  __kernel void filterOffsetStore(
                      //tracklets
                      __global const uint * trackletHitId1,
                      const uint nTracklets, const uint nLayerTriplets,
                      //hit data
                      __global const uint * hitEvent, __global const uint * hitLayer,
                      //output tracklet offst
                      __global uint * trackletOffsets)
    {
        size_t gid = get_global_id(0);

        if (gid >= nTracklets) {
          return;
        }

        uint hitId = trackletHitId1[gid];
        uint layerTriplet = hitLayer[hitId];
        uint event = hitEvent[hitId];

        if (gid > 0) {
            uint previousHitId = trackletHitId1[gid - 1];
            //this thread is the last one processing an element of this particular event or layer triplet
            if (layerTriplet != hitLayer[previousHitId] || event != hitEvent[previousHitId]) {
                trackletOffsets[event * nLayerTriplets + layerTriplet - 1] = gid;
            }
            // if this triplet is the last one, the offset has to be stored in the next (and usually the last position) of the offset buffer.
            if ((gid + 1) == nTracklets){
              // since layerTriplet starts by 1, for this particular case, we have the right position without substracting 1 to the layerTriplet.
              trackletOffsets[event * nLayerTriplets + layerTriplet] = gid;
            }
        } else {
            trackletOffsets[event * nLayerTriplets + layerTriplet - 1] = gid;
        }
        printf("\n");
    },
    cl_mem,
    cl_uint, cl_uint,
    cl_mem, cl_mem,
    cl_mem);


    // TODO I don't see how the offset list couldn't be monotonous since
    // event, layerTriplets and gid increase monotonically 
    KERNEL_CLASSP(filterOffsetMonotonizeStore,
                  oclDEFINES,

                  __kernel void filterOffsetMonotonizeStore(
                      __global uint * trackletOffsets, const uint nOffsets)
    {
        size_t gid = get_global_id(0);

        if (0 < gid && gid <= nOffsets) {
            if (trackletOffsets[gid] == 0) {
                trackletOffsets[gid] = trackletOffsets[gid - 1];
            }
        }
    }, cl_mem, cl_uint);
};

