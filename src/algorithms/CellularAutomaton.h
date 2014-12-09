#pragma once

#include <clever/clever.hpp>
#include <datastructures/TrackletCollection.h>
#include <datastructures/KernelWrapper.hpp>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

class CellularAutomaton : public KernelWrapper<CellularAutomaton>
{
public:
    CellularAutomaton(clever::context & context) :
        KernelWrapper(context),
        iteration(context),
        memset(context),
        memcpy(context),
        followerBasisCount(context),
        followerBasisStore(context),
        followerBestBasisStore(context),
        handlersStateStore(context),
        trackCollectionStore(context)
    {
        PLOG << "iteration WorkGroupSize: "
             << iteration.getWorkGroupSize() << std::endl
             << "memset WorkGroupSize: "
             << memset.getWorkGroupSize() << std::endl
             << "memcpy WorkGroupSize: "
             << memcpy.getWorkGroupSize() << std::endl
             << "followerBasisCount WorkGroupSize: "
             << followerBasisCount.getWorkGroupSize() << std::endl
             << "followerBestBasisStore WorkGroupSize: "
             << followerBestBasisStore.getWorkGroupSize() << std::endl
             << "handlersStateStore WorkGroupSize: "
             << handlersStateStore.getWorkGroupSize() << std::endl
             << "trackCollectionStore WorkGroupSize: "
             << trackCollectionStore.getWorkGroupSize() << std::endl;
    }

    //void
    std::tuple<const clever::vector<uint, 1>*, const clever::vector<uint, 1>*>
    run(const clever::vector<uint, 1> &tripletsBasis,
             const clever::vector<uint, 1> &tripletsFollowers,
             const clever::vector<float, 1> &tripletsPt,
             const clever::vector<uint, 1> &connectableTriplets,
             uint nThreads,
             bool printPROLIX) const;

    //TODO states could be uchars, nextLiving cells could be bitwise
    KERNEL_CLASS(iteration,
                 __kernel void iteration(
                     //input
                     const __global uint * const __restrict tripletsBasis,
                     const __global uint * const __restrict tripletsFollowers,
                     const __global uint * const __restrict currentStates,
                     //TODO ?
                     //__global uint * const __restrict currentLivingCells,
                     //output
                     __global uint * const __restrict nextStates,
                     __global uint * const __restrict livingCells,
                     //workload
                     const uint nTripletPairs)
    {
        const size_t tripletPair = get_global_id(0);

        if (tripletPair >= nTripletPairs) {
            return;
        }

        //const uint tripletIndexA = tripletA[gid];
        const uint tripletFollower = tripletsFollowers[tripletPair];
        const uint followerState = currentStates[tripletFollower];
        const bool sameStateTest = currentStates[tripletsBasis[tripletPair]] == followerState;
        //printf("BEFORE: %u %u", nextStates[tripletFollower],  followerState + sameStateTest);

        //uint old = atomic_max(&(nextStates[tripletFollower]), followerState + sameStateTest);
        atomic_max(&(nextStates[tripletFollower]), followerState + sameStateTest);
        if (sameStateTest) {
            //atomic_or(&(livingCells[tripletFollower]), sameStateTest);
            //nextStates[tripletFollower] = followerState + 1;
            livingCells[tripletFollower] = 1;
        }
        //printf("before: %u %u\n", nextStates[tripletFollower],  followerState + sameStateTest);

        /*
            printf("%lu, (%u %u): (%u %u) %d -> (old:%u %u) %u !!! %u %u\n",
                tripletPair,
                tripletsBasis[tripletPair], tripletFollower,
                currentStates[tripletsBasis[tripletPair]], followerState, sameStateTest,
                old, nextStates[tripletFollower],
                followerState + sameStateTest,
                a, old2);
        */

    },
    cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_uint);

    KERNEL_CLASS(memset,
                 __kernel void memset(
                     //input
                     __global uint * const __restrict memory,
                     const uint value,
                     //workload
                     const uint length)
    {
        const size_t gid = get_global_id(0);

        if (gid >= length) {
            return;
        }
        memory[gid] = value;
    },
    cl_mem, cl_uint,
    cl_uint);

    KERNEL_CLASS(memcpy,
                 __kernel void memcpy(
                     //input
                     const __global uint * const __restrict src,
                     //output
                     __global uint * const __restrict dst,
                     //workload
                     const uint length)
    {
        const size_t gid = get_global_id(0);

        if (gid >= length) {
            return;
        }

        dst[gid] = src[gid];
    },
    cl_mem,
    cl_mem,
    cl_uint);

    KERNEL_CLASS(followerBasisCount,
                 __kernel void followerBasisCount(
                     //input
                     const __global uint * const __restrict tripletsBasis,
                     const __global uint * const __restrict tripletsFollowers,
                     const __global uint * const __restrict tripletsStatus,
                     //output
                     __global uint * const __restrict followerBasisCount,
                     //workload
                     const uint nTripletPairs)
    {
        const size_t tripletPair = get_global_id(0);

        if (tripletPair >= nTripletPairs) {
            return;
        }
        //this could be stored on a vector to have the offsets already and do not need
        //to call atomic_inc again in the store phase
        const uint tripletFollower = tripletsFollowers[tripletPair];

        if (tripletsStatus[tripletsBasis[tripletPair]] + 1 != tripletsStatus[tripletFollower]) {
            return;
        }

        atomic_inc(&(followerBasisCount[tripletFollower]));
        //uint storeOffset = atomic_inc(&(followerBasisCount[tripletFollower]));
        //printf("STORE OFFSETA: TP-GPU#%lu [%u] %u -> %u\n", tripletPair, storeOffset + 1, tripletsBasis[tripletPair], tripletFollower);
    },
    cl_mem, cl_mem, cl_mem,
    cl_mem,
    cl_uint);

    KERNEL_CLASS(followerBasisStore,
                 __kernel void followerBasisStore(
                     //input
                     const __global uint * const __restrict tripletsBasis,
                     const __global uint * const __restrict tripletsFollowers,
                     const __global uint * const __restrict tripletsStatus,
                     const __global float * const __restrict tripletsPt,
                     //I/O, but the output is useless (will be the inclusive prefix sum actually)
                     __global uint * const __restrict followerBasisCountPrefixSum ,
                     //output
                     __global float * const __restrict ptDiff,
                     __global uint * const __restrict basisIndexes,
                     //workload
                     const uint nTripletPairs)
    {
        const size_t tripletPair = get_global_id(0);

        if (tripletPair >= nTripletPairs) {
            return;
        }

        const uint tripletBasis = tripletsBasis[tripletPair];
        const uint tripletFollower = tripletsFollowers[tripletPair];

        if (tripletsStatus[tripletsBasis[tripletPair]] + 1 != tripletsStatus[tripletFollower]) {
            return;
        }

        uint storeOffset = atomic_inc(&(followerBasisCountPrefixSum[tripletFollower]));
        //printf("STORE OFFSETA: #%lu [%u] %u %u\n", tripletPair, storeOffset, tripletFollower, tripletBasis);
        basisIndexes[storeOffset] = tripletBasis;
        ptDiff[storeOffset] = fabs(tripletsPt[tripletBasis] - tripletsPt[tripletFollower]);
    },

    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_uint);


    KERNEL_CLASS(followerBestBasisStore,
                 __kernel void followerBestBasisStore(
                     //input
                     const __global uint * const __restrict followerBasisCountPrefixSum,
                     const __global float * const __restrict followerBasisPtDifference,
                     const __global uint * const __restrict basisIndexes,
                     //output
                     __global uint * const __restrict followerBestBasis,
                     __global uint * const __restrict tripletIsBestBasis,
                     //workload
                     const uint nTriplets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTriplets) {
            return;
        }
        //printf("THREAD %d");
        // at the beginning no triplets are best basis, it is set here
        // rather than initializing and transferring a buffer.

        //const uint begin = followerBasisCountPrefixSum[tripletIndex];
        //const uint end = followerBasisCountPrefixSum[tripletIndex + 1];
        const uint notThread0 = tripletIndex > 0;
        //index = 0 => to 0 anyway;
        //index > 0 => shift one to left
        const uint offsetAsInExclusivePS = (tripletIndex - 1) * notThread0;
        //index = 0 => reads from 0, valid position, but then multiplies by 0 so it gets 0
        //index > 0 => read from the shifted position, multiplies by 1 hence no change
        const uint begin = followerBasisCountPrefixSum[offsetAsInExclusivePS] * notThread0;
        //index = 0 => reads from 0 + 0, so its end.
        //index > 0 => reads from the shifted position + 1
        const uint end = followerBasisCountPrefixSum[offsetAsInExclusivePS + 1 * notThread0];

        // otherwise it will get inside an infinite loop!
        if (begin == end) {
            return;
        }

        float minPt = followerBasisPtDifference[begin];
        uint minPtIndex = basisIndexes[begin];

        for (uint i = begin + 1; i < end; i++) {
            float ptDiff = followerBasisPtDifference[i];
            uint test = ptDiff < minPt;
            minPt = minPt * !test + ptDiff * test;
            minPtIndex =  minPtIndex * !test + basisIndexes[i] * test;
        }

        followerBestBasis[tripletIndex] = minPtIndex;
        tripletIsBestBasis[minPtIndex] = 1;
    },

    cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_uint);

    //TODO can this kernel and the previous be merged?
    KERNEL_CLASS(handlersStateStore,
                 __kernel void handlersStateStore(
                     //input
                     const __global uint * const __restrict tripletStates,
                     const __global uint * const __restrict tripletIsBestBasis,
                     //output
                     __global uint * const __restrict tripletHandlerStatesCount,
                     //workload
                     const uint nTriplets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTriplets) {
            return;
        }

        tripletHandlerStatesCount[tripletIndex] =
                (tripletIsBestBasis[tripletIndex] == 0) * tripletStates[tripletIndex];
    },

    cl_mem, cl_mem,
    cl_mem,
    cl_uint);

    KERNEL_CLASS(trackCollectionStore,
                 __kernel void trackCollectionStore(
                     //input
                     //const __global uint * const __restrict tripletStates,
                     const __global uint * const __restrict followerBestBasis,
                     const __global uint * const __restrict tripletHandlerStatesPrefixSum,
                     //output
                     __global uint * const __restrict trackCollection,
                     //workload
                     const uint nTriplets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTriplets) {
            return;
        }


        const uint storageOffset = tripletHandlerStatesPrefixSum[tripletIndex];
        const uint length = tripletHandlerStatesPrefixSum[tripletIndex + 1] - storageOffset;

        if (length == 0) {
            return;
        }
        uint basisIndex = tripletIndex;
        for (uint i = 0; i < length; i++) {
            trackCollection[storageOffset + i] = basisIndex;
            basisIndex = followerBestBasis[basisIndex];
        }
    },

    cl_mem, cl_mem,
    cl_mem,
    cl_uint);
};
