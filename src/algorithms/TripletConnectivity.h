#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/TrackletCollection.h>

using namespace clever;
using namespace std;

/*
 This class contains the infrastructure and kernel to compute the connectivity quantity for
 triplets. During this process the compatible tracklets which can form a complete track are
 counted and the number is stored with the triplet.

 Input:
 - buffer A holding Triplets to compute : read / write
 - buffer B holding Tripltes to check for connectivity ( can be the same as above ) : read only
 - range to start and end the connectivity search on buffer B

 possible todos:
 - improve performance by using local caching of the connectivity quantity
 - more complex tests ( fitted trajectory, theta / phi values of triptles )

 */
class TripletConnectivity: private boost::noncopyable
{
private:

    clever::context & m_ctx;

public:

    TripletConnectivity(clever::context & ctext);

    static std::string KERNEL_COMPUTE_EVT()
    {
        return "TripletConnectiviy_COMPUTE";
    }
    static std::string KERNEL_STORE_EVT()
    {
        return "TripletConnectiviy_STORE";
    }

    /*
     by default, only the outermost hits of both triplets are compared for compatibility.
     The tightPacking option can be set to true to compute the connectivity for overlapping


     */

    void run(TrackletCollection  const& trackletsBase,
             TrackletCollection  & trackletsFollowing,
             bool iterateBackwards = false, bool tightPacking = false) const;
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

    KERNEL_CLASS(tripletConnectivityTight,
                 __kernel void tripletConnectivityTight(
                     // tracklet base ( hit id, hit id, connectivity )
                     __global uint const* __restrict tripletBaseHit1,
                     __global uint const* tripletBaseHit2,
                     __global uint const* tripletBaseCon,

                     // tracklet following ( hit id, hit id, connectivity )
                     __global uint const* tripletFollowHit1,
                     __global uint const* tripletFollowHit2,
                     __global uint * tripletFollowCon,

                     const uint baseFirst, const uint baseLast
                 )
    {
        const size_t gid = get_global_id(0);
        for (size_t i = baseFirst; i <= baseLast; i++) {
            //printf("Doing %i\n" , gid );
            // the comparison has to be made criss / cross
            const bool connected = (tripletBaseHit1[i] == tripletFollowHit1[gid]) &&
                                   (tripletBaseHit2[i] == tripletFollowHit2[gid]);

            tripletFollowCon [gid] += connected * (tripletBaseCon[i] + 1);

        }

    },
    cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_uint, cl_uint);

};

