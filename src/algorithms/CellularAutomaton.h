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
        countFollowerBasis(context),
        storeFollowerBasis(context),
        determineFollowerBestBasis(context)
    {
    }
    
    void run(const clever::vector<uint, 1> &tripletsBasis,
             const clever::vector<uint, 1> &tripletsFollowers,
             const clever::vector<float, 1> &tripletsPt,
             const uint nThreads,
             bool printPROLIX) const;

    //TODO states could be uchars
    // nextLiving cells could bit level
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
        const uint stateFollower = currentStates[tripletFollower];
        const bool sameStateTest = currentStates[tripletsBasis[tripletPair]] == stateFollower;
        //printf("BEFORE: %u %u", nextStates[tripletFollower],  stateFollower + sameStateTest);
        uint old = atomic_max(&(nextStates[tripletFollower]), stateFollower + sameStateTest);
        atomic_or(&(livingCells[tripletFollower]), sameStateTest);
        //printf("before: %u %u\n", nextStates[tripletFollower],  stateFollower + sameStateTest);

    /*
        printf("%lu, (%u %u): (%u %u) %d -> (old:%u %u) %u !!! %u %u\n", 
            tripletPair, 
            tripletsBasis[tripletPair], tripletFollower,
            currentStates[tripletsBasis[tripletPair]], stateFollower, sameStateTest,
            old, nextStates[tripletFollower],
            stateFollower + sameStateTest,
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

    KERNEL_CLASS(countFollowerBasis,
        __kernel void countFollowerBasis(
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
  
        if (tripletsStatus[tripletsBasis[tripletPair]] + 1 != tripletsStatus[tripletFollower]){
            return;
        }

        atomic_inc(&(followerBasisCount[tripletFollower]));
        //uint storeOffset = atomic_inc(&(followerBasisCount[tripletFollower]));
        //printf("STORE OFFSETA: TP-GPU#%lu [%u] %u -> %u\n", tripletPair, storeOffset + 1, tripletsBasis[tripletPair], tripletFollower);
    },
    cl_mem, cl_mem, cl_mem,
    cl_mem,
    cl_uint);

    KERNEL_CLASS(storeFollowerBasis,
        __kernel void storeFollowerBasis(
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
        
        if (tripletsStatus[tripletsBasis[tripletPair]] + 1 != tripletsStatus[tripletFollower]){
            return;
        }
        
        uint storeOffset = atomic_inc(&(followerBasisCountPrefixSum[tripletFollower]));
        //printf("STORE OFFSETA: #%lu [%u] %u %u\n", tripletPair*0,  0*storeOffset, tripletFollower, tripletBasis);
        basisIndexes[storeOffset] = tripletBasis;
        ptDiff[storeOffset] = fabs(tripletsPt[tripletBasis] - tripletsPt[tripletFollower]);
    },
    
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_uint);


    KERNEL_CLASS(determineFollowerBestBasis,
        __kernel void determineFollowerBestBasis(
            //input
            const __global uint * const __restrict followerBasisCountPrefixSum,
            const __global float * const __restrict followerBasisPtDifference,
            const __global uint * const __restrict basisIndexes,
            //output
            __global uint * const __restrict followerBestBasis,
            __global uint * const __restrict tripletIsBestBasisForFollower,
            //workload
            const uint nTriplets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTriplets) {
            return;
        }

        const uint begin = followerBasisCountPrefixSum[tripletIndex];
        const uint end = followerBasisCountPrefixSum[tripletIndex + 1];
        
        if(begin == end){
            return;
        }

        float minPt = followerBasisPtDifference[begin];
        uint minPtIndex = basisIndexes[begin];
        
        for(uint i = begin + 1; i < end; i++){
            float ptDiff = followerBasisPtDifference[i];
            uint test = ptDiff < minPt;
            minPt = minPt * !test + ptDiff * test;
            minPtIndex =  minPtIndex * !test + basisIndexes[i] * test;
        }
        
        followerBestBasis[tripletIndex] = minPtIndex;
        tripletIsBestBasisForFollower[minPtIndex] = 1;
    },
    
    cl_mem, cl_mem, cl_mem,
    cl_mem, cl_mem,
    cl_uint);
};
