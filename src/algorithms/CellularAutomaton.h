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
        memset(context)
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
            const __global uint * const __restrict tripletA,
            const __global uint * const __restrict tripletB,
            const __global uint * const __restrict currentState,
            //TODO ?
            //__global uint * const __restrict currentLivingCells,
            //output
            __global uint * const __restrict nextState,
            __global uint * const __restrict livingCells,
            //workload
            const uint nTripletPairs)
    {
        const size_t gid = get_global_id(0);

        if (gid >= nTripletPairs) {
            return;
        }

        //const uint tripletIndexA = tripletA[gid];
        const uint tripletIndexB = tripletB[gid];
        const uint stateB = currentState[tripletIndexB];
        const uint test = currentState[tripletA[gid]] == stateB;
        //TODO surround atomics by if?
        if(test){
            const uint old = atomic_max(&(nextState[tripletIndexB]), stateB + test);
            if (old < stateB + test){
                atomic_or(&(livingCells[tripletIndexB]), test);
                //updatedTripletsPrefixSum[tripletIndexB] = 1;
            }
        }
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
};
