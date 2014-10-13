#include "CellularAutomaton.h"
#include <memory>
#include <limits>

void CellularAutomaton::run(const clever::vector<uint, 1> &tripletsBasis,
                            const clever::vector<uint, 1> &tripletsFollowers,
                            const clever::vector<float, 1> &tripletsPt,
                            const uint nThreads,
                            bool printPROLIX) const
{
    LOG << std::endl << "BEGIN CellularAutomaton" << std::endl;
    
    if(((PROLIX) && printPROLIX)){
        LOG << std::endl << "TripletPairs:" << std::endl;
        std::vector<uint> vBasis(tripletsBasis.get_count());
        transfer::download(tripletsBasis, vBasis, ctx);

        std::vector<uint> vFollowers(tripletsFollowers.get_count());
        transfer::download(tripletsFollowers, vFollowers, ctx);
        
        for (uint i = 0; i < vBasis.size(); i++) {
            PLOG << "TP-CPU# " << i << " " << vBasis[i] << " -> " << vFollowers[i] << std::endl;
        }
    }
    const uint nTripletPairs = tripletsBasis.get_count();
    const uint nTriplets = tripletsPt.get_count();
    const uint nGroupsForTriplets = (uint) std::max(1.0f, ceil(((float) nTriplets) / nThreads));
    const uint nGroupsForPairs = (uint) std::max(1.0f, ceil(((float) nTripletPairs) / nThreads));

    clever::vector<uint, 1> * tripletsStates = new clever::vector<uint, 1>(1, nTriplets, ctx);
    clever::vector<uint, 1> * tripletNextState = new clever::vector<uint, 1>(1, nTriplets, ctx);
    clever::vector<uint, 1> * const livingCells = new clever::vector<uint, 1>(nTriplets + 1, ctx);
    PrefixSum prefixSum(ctx);
    
    cl_event evt;
    uint aliveCells;
    
    LOG << std::endl << "Runing CA, forward phase." << std::endl;
    uint iterationCount = 0;
    do{
        evt = memset.run(livingCells->get_mem(), 0, nTriplets, range(nGroupsForTriplets * nThreads), range(nThreads));
        CellularAutomaton::events.push_back(evt);
        evt = iteration.run(
                //input
                tripletsBasis.get_mem(),
                tripletsFollowers.get_mem(),
                tripletsStates->get_mem(),
                //output
                tripletNextState->get_mem(),
                livingCells->get_mem(),
                //workload
                nTripletPairs,
                //configuration
                range(nGroupsForPairs * nThreads),
                range(nThreads));
        CellularAutomaton::events.push_back(evt);
        //TODO do not use one boolean for each triplet but group them, the popcount might be enough
        //even the place of storage doesn't matter as the only important thing is having zero or not.

        // prefix sum of living cells -> stop if prefix sum == 0;
        evt = prefixSum.run(
            livingCells->get_mem(),
            livingCells->get_count(),
            nThreads,
            CellularAutomaton::events);
        CellularAutomaton::events.push_back(evt);
        transfer::downloadScalar(*livingCells, aliveCells, ctx, true, livingCells->get_count() - 1, 1, &evt);

        std::swap(tripletsStates, tripletNextState);

        PLOG << "Counted a total of " << aliveCells << " alive cells."<< std::endl;
        iterationCount++;
    }while(aliveCells > 0);


    if(((PROLIX) && printPROLIX)){
        std::vector<uint> vTBasis(tripletsBasis.get_count());
        transfer::download(tripletsBasis, vTBasis, ctx);
        
        std::vector<uint> vTFollowers(tripletsFollowers.get_count());
        transfer::download(tripletsFollowers, vTFollowers, ctx);
        
        std::vector<uint> vCurrent(tripletsStates->get_count());
        transfer::download(*tripletsStates, vCurrent, ctx);
        
        std::vector<uint> vNext(tripletNextState->get_count());
        transfer::download(*tripletNextState, vNext, ctx);
        
        std::vector<uint> vLiving(livingCells->get_count());
        transfer::download(*livingCells, vLiving, ctx);


        for (uint i = 0; i < vTBasis.size(); i++) {
            PLOG << "CA# " << i
                 << " B " << vTBasis[i]
                 << "-> F " << vTFollowers[i]
                 << " [" << vLiving[vTBasis[i]]
                 << ", " << vLiving[vTFollowers[i]]
                 << "] " << vCurrent[vTBasis[i]]
                 << " -> " << vNext[vTBasis[i]]
                 << " " << vCurrent[vTFollowers[i]]
                 << " -> " << vNext[vTFollowers[i]]
                 << std::endl;
        }
    }

    LOG << std::endl << "Runing CA, backward phase." << std::endl;
    LOG << std::endl << "Counting basis for a given follower." << "triplets: "<< nTriplets << "triplet pairs: " << nTripletPairs << std::endl;

    clever::vector<uint, 1> * const followerBasisCountPrefixSum = new clever::vector<uint, 1>(0, nTriplets + 1, ctx);
    evt = countFollowerBasis.run(
        //input
        tripletsBasis.get_mem(),
        tripletsFollowers.get_mem(),
        tripletsStates->get_mem(),
        //output
        followerBasisCountPrefixSum->get_mem(),
        //workload
        nTripletPairs,
        //config
        range(nGroupsForPairs * nThreads),
        range(nThreads));
    CellularAutomaton::events.push_back(evt);

    if(((PROLIX) && printPROLIX)){
        LOG << std::endl << "Basis for a given follower:" << std::endl;
        std::vector<uint> vfollowerBasisCountPrefixSum(followerBasisCountPrefixSum->get_count());
        transfer::download(*followerBasisCountPrefixSum, vfollowerBasisCountPrefixSum, ctx);

        std::vector<uint> vTFollowers(tripletsFollowers.get_count());
        transfer::download(tripletsFollowers, vTFollowers, ctx);
        
        for (uint i = 0; i < vfollowerBasisCountPrefixSum.size(); i++) {
            PLOG << i << " " << vfollowerBasisCountPrefixSum[i] << std::endl;
        }
    }
    
    evt = prefixSum.run(
        followerBasisCountPrefixSum->get_mem(),
        followerBasisCountPrefixSum->get_count(),
        nThreads,
        CellularAutomaton::events);
    CellularAutomaton::events.push_back(evt);
    
    uint followerBasisCountPrefixSumTotal;
    transfer::downloadScalar(*followerBasisCountPrefixSum, followerBasisCountPrefixSumTotal, ctx, true, followerBasisCountPrefixSum->get_count() - 1, 1, &evt);
    
    LOG << std::endl << "Number of triplets with basis: " << followerBasisCountPrefixSumTotal << std::endl;

    if(((PROLIX) && printPROLIX)){
        LOG << std::endl << "Prefix sum of basis count:" << std::endl;
        std::vector<uint> vfollowerBasisCountPrefixSum(followerBasisCountPrefixSum->get_count());
        transfer::download(*followerBasisCountPrefixSum, vfollowerBasisCountPrefixSum, ctx);
        for (uint i = 0; i < vfollowerBasisCountPrefixSum.size(); i++) {
            PLOG << i << ' ' << vfollowerBasisCountPrefixSum[i] << std::endl;
        }

        PLOG << "HOW THE STORAGE WILL LOOK LIKE " << std::endl;

        for (uint triplet = 0; triplet < nTriplets; triplet++){
            for(uint basisOffset = vfollowerBasisCountPrefixSum[triplet]; basisOffset < vfollowerBasisCountPrefixSum[triplet + 1]; basisOffset++){
                PLOG << triplet << ' ' << basisOffset << std::endl;
            }
        }
    }


    //anotate Pt and indices of basis for a given triplet in adyacent positions
    clever::vector<uint, 1> * const followerBasisIndices = new clever::vector<uint, 1>(followerBasisCountPrefixSumTotal, ctx);
    clever::vector<float, 1> * const followerBasisPtDiff = new clever::vector<float, 1>(followerBasisCountPrefixSumTotal, ctx);
    
    clever::vector<uint, 1> * followerBasisCountPrefixSumForStorage = new clever::vector<uint, 1>(followerBasisCountPrefixSum->get_count(), ctx);

    evt = memcpy.run(followerBasisCountPrefixSum->get_mem(),
        followerBasisCountPrefixSumForStorage->get_mem(),
        followerBasisCountPrefixSumForStorage->get_count(),
        range(nGroupsForTriplets * nThreads), range(nThreads));
    CellularAutomaton::events.push_back(evt);


    evt = storeFollowerBasis.run(
        //input
        tripletsBasis.get_mem(),
        tripletsFollowers.get_mem(),
        tripletsStates->get_mem(),
        tripletsPt.get_mem(),
        //I/O
        followerBasisCountPrefixSumForStorage->get_mem(),
        //out
        followerBasisPtDiff->get_mem(),
        followerBasisIndices->get_mem(),
        //workload
        nTripletPairs,
        //configuration
        range(nGroupsForTriplets * nThreads),
        range(nThreads));
    CellularAutomaton::events.push_back(evt);
    
    delete followerBasisCountPrefixSumForStorage;

    if(((PROLIX) && printPROLIX)){
        LOG << std::endl << "Basis index and pt difference:" << std::endl;
        std::vector<uint> vFollowerBasisIndices(followerBasisIndices->get_count());
        transfer::download(*followerBasisIndices, vFollowerBasisIndices, ctx);
        
        std::vector<float> vFollowerBasisPtDiff(followerBasisPtDiff->get_count());
        transfer::download(*followerBasisPtDiff, vFollowerBasisPtDiff, ctx);
        
        std::vector<uint> vTripletsStates(tripletsStates->get_count());
        transfer::download(*tripletsStates, vTripletsStates, ctx);
        
        std::vector<uint> vfollowerBasisCountPrefixSum(followerBasisCountPrefixSum->get_count());
        transfer::download(*followerBasisCountPrefixSum, vfollowerBasisCountPrefixSum, ctx);
        
        for (uint triplet = 0; triplet < nTriplets; triplet++){
            PLOG << "Follower: " << triplet << " state " <<  vTripletsStates[triplet] << std::endl;
            for(uint basisOffset = vfollowerBasisCountPrefixSum[triplet]; basisOffset < vfollowerBasisCountPrefixSum[triplet + 1]; basisOffset++){                
                PLOG << '\t' << " [" << basisOffset << "] "
                     << "Basis: " << vFollowerBasisIndices[basisOffset]
                     << " state " << vTripletsStates[vFollowerBasisIndices[basisOffset]]
                     << ' ' << vFollowerBasisPtDiff[basisOffset]
                     << std::endl;
            }
            PLOG << "------------------------------------" << std::endl;
        }
    }

    
    const uint max_uint = std::numeric_limits<uint>::max();
    clever::vector<uint, 1> * const followerBestBasisIndices = new clever::vector<uint, 1>(max_uint, nTriplets, ctx);
    clever::vector<uint, 1> * const tripletIsBestBasisForFollower = new clever::vector<uint, 1>(nTriplets, ctx);

    evt = determineFollowerBestBasis.run(
        //input
        followerBasisCountPrefixSum->get_mem(),
        followerBasisPtDiff->get_mem(),
        followerBasisIndices->get_mem(),
        //ouput
        followerBestBasisIndices->get_mem(),
        tripletIsBestBasisForFollower->get_mem(),
        //workload
        nTriplets,
        //config
        range(nGroupsForTriplets * nThreads),
        range(nThreads));
    CellularAutomaton::events.push_back(evt);

    delete tripletsStates;
    delete tripletNextState;
    delete livingCells;
    delete followerBasisCountPrefixSum;
    delete followerBasisIndices;
    delete followerBasisPtDiff;

    // get the state of every basis triplet -> triplets that aren't best basis are outer beginnings of tracks
    // calculate prefixum of the states of triplets that aren't best basis -> we get th 
    // count them, get the prefix sum
    // store the tracks one after another given the offsets and the 
    LOG << std::endl << "END CellularAutomaton" << std::endl;
}
