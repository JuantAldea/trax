#pragma once

#include <tuple>

#include <clever/clever.hpp>
#include <datastructures/TrackletCollection.h>
#include <datastructures/HitCollection.h>
#include <datastructures/KernelWrapper.hpp>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

/*
 possible todos:
 - improve performance by using local caching of the connectivity quantity
 - more complex tests ( fitted trajectory, theta / phi values of triptles )

 */

class TripletConnectivityTight : public KernelWrapper<TripletConnectivityTight>
{

public:
    TripletConnectivityTight(clever::context & context) :
        KernelWrapper(context),
        tripletEtaCalculatorStore(context),
        tripletConnectivityTightCount(context),
        tripletConnectivityTightStore(context),
        streamCompactionGetValidIndexesStore(context),
        streamCompactionFilter3StreamsStore(context),
        predicateInversionInPlaceStore(context),
        predicateInversionStore(context)
    {
        PLOG << "tripletConnectivityTightCount WorkGroupSize: "
             << tripletConnectivityTightCount.getWorkGroupSize() << std::endl
             << "tripletConnectivityTightStore WorkGroupSize: "
             << tripletConnectivityTightStore.getWorkGroupSize() << std::endl;
    }

    std::tuple<clever::vector<uint, 1>*, clever::vector<uint, 1>*, clever::vector<uint, 1>*, clever::vector<uint, 1>*>
    run(const HitCollection &hits, TrackletCollection &tracklets, const float dEtaCut, const uint nThreads, bool printPROLIX = false) const;
    

    KERNEL_CLASS(tripletEtaCalculatorStore,
                 __kernel void tripletEtaCalculatorStore(
                    //input
                     __global const float * const __restrict hitX,
                     __global const float * const __restrict hitY,
                     __global const float * const __restrict hitZ,
                     __global const uint * const __restrict tripletInnerHitId,
                     __global const uint * const __restrict tripletOuterHitId,
                     //output
                     __global float * const __restrict tripletEta,
                     //workload
                     const uint nTracklets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTracklets) {
            return;
        }
        
        const uint hitInnerIndex = tripletInnerHitId[tripletIndex];
        const uint hitOuterIndex = tripletOuterHitId[tripletIndex];

        float3 hitInner = (float3)(hitX[hitInnerIndex], hitY[hitInnerIndex], hitZ[hitInnerIndex]);
        float3 hitOuter = (float3)(hitX[hitOuterIndex], hitY[hitOuterIndex], hitZ[hitOuterIndex]);
        float3 p = (float3)(hitOuter - hitInner);
        
        const float t = p.z / sqrt(p.x * p.x + p.y * p.y);
        
        tripletEta[tripletIndex] = asinh(t);
    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
    cl_mem,
    cl_uint);


    // TODO Incorporate layer information.
    KERNEL_CLASS(tripletConnectivityTightCount,
                 __kernel void tripletConnectivityTightCount(
                     //input
                     __global const uint * const __restrict hitsBasisH1,
                     __global const uint * const __restrict hitsBasisH2,
                     __global const uint * const __restrict hitsFollowersH0,
                     __global const uint * const __restrict hitsFollowersH1,
                     __global const float * const __restrict tripletsEta,
                     const float dEtaCut,
                     //output
                     //connectivityCount holds the number of followers
                     //for a given triplet.
                     __global uint * const __restrict connectivityCount,
                     __global uint * const __restrict connectivityOracle,
                     //workload
                     const uint nTracklets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTracklets) {
            return;
        }

        const uint hitIndexInner = hitsBasisH1[tripletIndex];
        const uint hitIndexOuter = hitsBasisH2[tripletIndex];
        const float localTripletEta = tripletsEta[tripletIndex];

        uint connectivityCountLocal = 0;
        for (uint i = 0; i < nTracklets; i++) {
            const bool test = (hitIndexInner == hitsFollowersH0[i])
                            * (hitIndexOuter == hitsFollowersH1[i])
                            * (fabs(localTripletEta - tripletsEta[i]) < dEtaCut);
            
            connectivityCountLocal += test;

            //protecting this atomic_or with the if reduces the kernel time form 60ms to 16
            if(test){
                //atomic_or(&connectivityOracle[i / sizeof(uint)], test << (i % sizeof(uint)));
                //atomic_or(&connectivityOracle[i], test);
                //given that this will always set to 1, is the atomic really needed?
                atomic_or(&connectivityOracle[i], 1);
                //connectivityOracle[i] = 1;
            }
        }

        connectivityCount[tripletIndex] = connectivityCountLocal;
        
        //atomic_or(&connectivityOracle[tripletIndex / sizeof(uint)], (connectivityCountLocal > 0) << (tripletIndex % sizeof(uint)));
        if(connectivityCountLocal){
            //atomic_or(&connectivityOracle[tripletIndex], connectivityCountLocal > 0);
            atomic_or(&connectivityOracle[tripletIndex], 1);
            //connectivityOracle[tripletIndex] = 1;
        }
    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_float,
    cl_mem, cl_mem,
    cl_uint);

    KERNEL_CLASS(tripletConnectivityTightStore,
                 __kernel void tripletConnectivityTightStore(
                    //input
                     __global const uint * const __restrict hitsBasisH1,
                     __global const uint * const __restrict hitsBasisH2,
                     __global const uint * const __restrict hitsFollowersH0,
                     __global const uint * const __restrict hitsFollowersH1,
                     __global const float * const __restrict tripletsEta,
                     const float dEtaCut,
                     __global const uint * const __restrict connectivityPrefixSum,
                     //output
                     __global uint * const __restrict baseTriplet,
                     __global uint * const __restrict followerTriplet,
                     //work load
                     const uint nTracklets)
    {
        const size_t tripletIndex = get_global_id(0);

        if (tripletIndex >= nTracklets) {
            return;
        }

        const uint hitIndexInner = hitsBasisH1[tripletIndex];
        const uint hitIndexOuter = hitsBasisH2[tripletIndex];
        const float localTripletEta = tripletsEta[tripletIndex];

        uint storeOffset = connectivityPrefixSum[tripletIndex];
        const uint storeOffsetNextTriplet = connectivityPrefixSum[tripletIndex + 1];
        
        //connectivityOracle not needed, vality given by storeOffsets difference
        for (uint i = 0; (i < nTracklets) * (storeOffset < storeOffsetNextTriplet); i++) {
            const bool test = (hitIndexInner == hitsFollowersH0[i]) 
                              * (hitIndexOuter == hitsFollowersH1[i])
                              * (fabs(localTripletEta - tripletsEta[i]) < dEtaCut);
            // the same position will be overwritten until it is valid
            // meaning no thread divergence here.
            baseTriplet[storeOffset] = tripletIndex;
            followerTriplet[storeOffset] = i;
            storeOffset += test;
        }
    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_float, cl_mem,
    cl_mem, cl_mem,
    cl_uint);
    

    //This filtering uses the prefix sums, its not the quickest way
    //of filtering by predicate but it is stable ie, it preserves
    //the original ordering between elements
    // a quicker but non-stable option could be made using warp aggregated
    // atomics.
    KERNEL_CLASS(streamCompactionGetValidIndexesStore,
                 __kernel void streamCompactionGetValidIndexesStore(
                    //input
                    __global const uint * const __restrict predicatePrefixSum,
                    //output
                    __global uint * const __restrict validIndexes,
                    //work load
                    const uint streamLength)
    {
        const size_t gid = get_global_id(0);

        if(gid >= streamLength) {
            return;
        }
        
        const uint localPredicatePrefixSum = predicatePrefixSum[gid];
        if(localPredicatePrefixSum != predicatePrefixSum[gid + 1]){
            validIndexes[localPredicatePrefixSum] = gid;
        }
    },
    cl_mem,
    cl_mem,
    cl_uint);

    KERNEL_CLASS(streamCompactionFilter3StreamsStore,
                __kernel void streamCompactionFilter3StreamsStore(
                    //input
                    __global const uint * const __restrict stream1,
                    __global const uint * const __restrict stream2,
                    __global const uint * const __restrict stream3,
                    __global const uint * const __restrict indexes,
                    //output
                    __global uint * const __restrict filteredStream1,
                    __global uint * const __restrict filteredStream2,
                    __global uint * const __restrict filteredStream3,
                    //work load
                    const uint nIndexes)
    {
        const size_t gid = get_global_id(0);

        if (gid >= nIndexes) {
            return;
        }
        
        //TODO this is an stupid and naive way of doing this
        //probably it's even better to call the kernel three
        //times as it will trash the caches less.
        // TO BE MEASURED
        filteredStream1[gid] = stream1[indexes[gid]];
        filteredStream2[gid] = stream2[indexes[gid]];
        filteredStream3[gid] = stream3[indexes[gid]];
    },
    cl_mem, cl_mem, cl_mem, cl_mem, 
    cl_mem, cl_mem, cl_mem,
    cl_uint);

    KERNEL_CLASS(predicateInversionInPlaceStore,
                __kernel void predicateInversionInPlaceStore(
                    //input
                    __global uint * const predicate,
                    //output
                    const uint nValues)
    {
        const size_t gid = get_global_id(0);

        if (gid >= nValues) {
            return;
        }
        
        predicate[gid] = !predicate[gid];
    },
    cl_mem,
    cl_uint);

    KERNEL_CLASS(predicateInversionStore,
                __kernel void predicateInversionStore(
                    //input
                    __global const uint * const __restrict predicate,
                    //output
                    __global uint * const __restrict invPredicate,
                    const uint nValues)
    {
        const size_t gid = get_global_id(0);

        if (gid >= nValues) {
            return;
        }
        
        invPredicate[gid] = !predicate[gid];
    },
    cl_mem,
    cl_mem,
    cl_uint);
};



/*
    KERNEL_CLASS(tripletConnectivityWide2,
                 __kernel void tripletConnectivityWide2(
                     // tracklet base (hit id, connectivity)
                     __global const uint * tripletBaseHit,
                     __global uint * tripletBaseConnectityDegree,
                     // tracklet following (hit id, connectivity)
                     __global const uint * tripletFollowerHit,
                     __global const uint * layerOffsets,
                     __global const uint layerOffsetsLen,
                     __global const uint * hitLayer,
                     const uint followerFirstIndex, const uint followerLastIndex
                 )
    {
        const size_t hitIndex = get_global_id(0);
        // try to not to thrash caches if this index should be skipped,
        // take an hit that should be nearby.
        // TODO jaldeear would be better to get the last know valid of the previous to the last seeding layer set.
        const uint defaultHitIndex = layerOffsets[layerOffsetsLen - 1];

        // the hit layer offset is the i that verifies layerOffset[i] <= hitIndex < layerOffsets[i + 1]
        // test it branchlessly
        uint myLayerIndex = 0;
        for (size_t i = 0; i < layerOffsetsLen - 1; i++) {
            myLayerIndex += i * (layerOffsets[i] <= hitIndex) * (hitIndex < layerOffsets[i + 1]);
        }

        // if myLayerIndex == 0 at this point there are three options:
        //  a) the seeding layer is 0
        //  b) the hit is on the last seeding layer (mind the -1 in the loop condition)
        //  c) hitIndex is larger than the amount of hits
        // Any hitIndex larger than the begin of the last seeding layer is not conectable
        // so it doesn't matter. Non existent hitIndex can be treat the same way.
        // We need to know the which case it is:
        //  a) do nothing
        //  b, c) set layer index out of bounds
        myLayerIndex += layerOffsetsLen * (layerOffsets[layerOffsetsLen - 1] <= hitIndex);
        // myLayerIndex == layerOffsetsLen --> we don't care about this index.
        const bool skipHitIndex = myLayerIndex == layerOffsetsLen;


        // the first compatible layer for a given base triplet verifies that
        // the last base and first follower hits belong to the same layer.
        // Doesn't make any sense to compare with triplets originated on the
        // inner seeding layers, hence the loop starts at 1

        const uint baseHit = tripletBaseHit[!skipHitIndex * hitIndex + skipHitIndex * defaultIndex];
        const uint baseHitLayer = layerHits[baseHit];

        uint firstCompatibleLayerOffsetIndex = 0;
        for (size_t i = 1; i < layerOffsetsLen; i++) {
            // use the first hit of every seeding layer set to find the beginning of the
            // compatible range.
            firstCompatibleLayerOffsetIndex += i * (hitLayer[layerOffsets[i]] == baseHitLayer);
        }

        // if firstCompatibleLayer == 0:
        //  a) the hit belongs to the outer seeding layer set -> do nothing.
        //  b) there are no triplets originated in a compatible seeding set -> do nothing
        // TODO jaldeaar: deal with b), quite unlikely though.
        const searchOffsetBegin = layerOffsets[firstCompatibleLayerOffsetIndex];
        const bool compatibleLayerIsTheLastLayer = (firstCompatibleLayerOffsetIndex + 1) == layerOffsetsLen;
        // if the compatible seeding layer for this hit is the last we don't have the offSet in the array
        // so we take the number of triplets as upper bound
        // if compatibleLayerIsTheLastLayer == 1: layersOffset does not go out of bounds since it is adding 0, and hopefully minimizes cache failures.
        // if compatibleLayerIsTheLastLayer == 0: layerOffset index is increased by 1 so it looks on the next position.
        const searchOffsetEnd = !compatibleLayerIsTheLastLayer * layerOffsets[firstCompatibleLayerOffsetIndex + !compatibleLayerIsTheLastLayer]
                                + compatibleLayerIsTheLastLayer * nTriplets;


        uint tripletConnectivityAcc = 0;
        //TODO jaldeaar: Investigate branch divergence originated here.
        for (size_t i = searchOffsetBegin; i < searchOffsetEnd; i++) {
            const bool connected = (baseHit == tripletFollowerHit[i]);
            tripletConnectivityAcc += connected;
        }

        if (!skipHitIndex){
            tripletBaseConnectityDegree[hitIndex] = tripletConnectivityAcc;
        }
    },
    cl_mem, cl_mem, cl_uint,
    cl_mem, cl_uint, cl_uint);
*/
/*
    KERNEL_CLASS(tripletConnectivityWide2,
                 __kernel void tripletConnectivityWide2(
                     // tracklet base (hit id, connectivity)
                     __global const uint * tripletBaseHit,
                     __global uint * tripletBaseConnectityDegree,
                     // tracklet following (hit id, connectivity)
                     __global const uint * tripletFollowerHit,
                     __global const uint * layerOffsets,
                     __global const uint layerOffsetsLen,
                     __global const uint * hitLayer,
                     __global const uint nTriplets
                 )
    {
        const size_t hitIndex = get_global_id(0);

        if (hitIndex >= nTriplets) {
            return;
        }

        const uint baseHit = tripletBaseHit[hitIndex];
        const uint baseHitLayer = hitLayer[baseHit];

        uint firstCompatibleLayerOffsetIndex = 0;
        for (size_t i = 1; i < layerOffsetsLen; i++) {
            // use the first hit of every seeding layer set to find the beginning of the
            // compatible range.
            firstCompatibleLayerOffsetIndex += i * (hitLayer[layerOffsets[i]] == baseHitLayer);
        }

        // if firstCompatibleLayer == 0:
        //  a) the hit belongs to the outer seeding layer set -> do nothing.
        //  b) there are no triplets originated in a compatible seeding set -> do nothing
        // TODO jaldeaar: deal with b), quite unlikely though.
        const uint searchOffsetBegin = layerOffsets[firstCompatibleLayerOffsetIndex];
        const bool compatibleLayerIsTheLastLayer = (firstCompatibleLayerOffsetIndex + 1) == layerOffsetsLen;
        // if the compatible seeding layer for this hit is the last we don't have the offSet in the array
        // so we take the number of triplets as upper bound
        // if compatibleLayerIsTheLastLayer == 1: layersOffset does not go out of bounds since it is adding 0, and hopefully minimizes cache failures.
        // if compatibleLayerIsTheLastLayer == 0: layerOffset index is increased by 1 so it looks on the next position.
        const uint searchOffsetEnd = !compatibleLayerIsTheLastLayer * layerOffsets[firstCompatibleLayerOffsetIndex + !compatibleLayerIsTheLastLayer]
                                     + compatibleLayerIsTheLastLayer * nTriplets;


        uint tripletConnectivityAcc = 0;
        //TODO jaldeaar: Investigate branch divergence originated here.
        for (size_t i = searchOffsetBegin; i < searchOffsetEnd; i++) {
            const bool connected = (baseHit == tripletFollowerHit[i]);
            tripletConnectivityAcc += connected;
        }

        tripletBaseConnectityDegree[hitIndex] = tripletConnectivityAcc;
    },
    cl_mem, cl_mem, cl_uint,
    cl_mem, cl_uint, cl_uint);

    KERNEL_CLASS(tripletConnectivityWide,
                 __kernel void tripletConnectivityWide(
                     // tracklet base (hit id, connectivity)
                     __global const uint * tripletBaseHit1,
                     __global const uint * tripletBaseCon,
                     // tracklet following (hit id, connectivity)
                     __global uint * tripletFollowHit1,
                     __global uint * tripletFollowCon,
                     const uint baseFirst, const uint baseLast
                 )
    {
        const size_t gid = get_global_id(0);

        uint tripletConnectivityAcc = 0;
        uint tripletBaseHit = tripletBaseHit1[gid];

        for (size_t i = baseFirst; i <= baseLast; i++) {
            const bool connected = (tripletBaseHit == tripletFollowHit1[i]);
            tripletConnectivityAcc += connected * (tripletBaseCon[i] + 1);
        }

        tripletFollowCon[gid] = tripletConnectivityAcc;
    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_uint, cl_uint);
*/