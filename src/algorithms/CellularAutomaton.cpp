#include "CellularAutomaton.h"

void CellularAutomaton::run(const clever::vector<uint, 1> &tripletsBasis,
                            const clever::vector<uint, 1> &tripletsFollowers,
                            const clever::vector<float, 1> &tripletsPt,
                            const uint nThreads,
                            bool printPROLIX) const
{
    LOG << std::endl << "BEGIN CellularAutomaton" << std::endl;
    
    const uint nTripletPairs = tripletsBasis.get_count();
    const uint nTriplets = tripletsPt.get_count();
    const uint nGroupsForTriplets = (uint) std::max(1.0f, ceil(((float) nTriplets) / nThreads));
    const uint nGroupsForPairs = (uint) std::max(1.0f, ceil(((float) nTripletPairs) / nThreads));

    clever::vector<uint, 1> * tripletCurrentState = new clever::vector<uint, 1>(1, nTriplets, ctx);
    clever::vector<uint, 1> * tripletNextState = new clever::vector<uint, 1>(1, nTriplets, ctx);
    clever::vector<uint, 1> * const livingCells = new clever::vector<uint, 1>(nTriplets + 1, ctx);
    PrefixSum prefixSum(ctx);
    
    cl_event evt;
    uint aliveCells;
    
    do{
        evt = memset.run(livingCells->get_mem(), 0, nTriplets, range(nGroupsForTriplets * nThreads), range(nThreads));
        CellularAutomaton::events.push_back(evt);

        evt = iteration.run(
                //input
                tripletsBasis.get_mem(),
                tripletsFollowers.get_mem(),
                tripletCurrentState->get_mem(),
                //output
                tripletNextState->get_mem(),
                livingCells->get_mem(),
                //workload
                nTripletPairs,
                //configuration
                range(nGroupsForPairs * nThreads),
                range(nThreads));
        CellularAutomaton::events.push_back(evt);
        
        evt = prefixSum.run(
            livingCells->get_mem(),
            livingCells->get_count(),
            nThreads,
            CellularAutomaton::events);
        CellularAutomaton::events.push_back(evt);
        transfer::downloadScalar(*livingCells, aliveCells, ctx, true, livingCells->get_count() - 1, 1, &evt);
        std::swap(tripletCurrentState, tripletNextState);
        PLOG << "Counted a total of" << aliveCells << " alive cells."<< std::endl;
    }while(aliveCells > 0);
    
    // prefix sum of living cells -> stop if prefix sum == 0;


    //Go backwards (B->C) selecting the triplet that maximizes State (length) and minimizes the Pt difference
    //  mark triplets best father as they are followers-> increase their number of parents by 1
    //  unmark those that were followers as they have turned into basis -> decrease their number of parents by 1
    //
    // get the state of every basis triplet -> triplets with 0 parents are track parents
    // count them, get the prefix sum
    // store the tracks one after another given the offsets and the 
    std::vector<uint> vTBasis(tripletsBasis.get_count());
    std::vector<uint> vTFollowers(tripletsFollowers.get_count());
    transfer::download(tripletsBasis, vTBasis, ctx);
    transfer::download(tripletsFollowers, vTFollowers, ctx);

    if(((PROLIX) && printPROLIX)){
        std::vector<uint> vNext(tripletNextState->get_count());
        std::vector<uint> vCurrent(tripletCurrentState->get_count());
        std::vector<uint> vLiving(livingCells->get_count());

        transfer::download(*tripletNextState, vNext, ctx);
        transfer::download(*tripletCurrentState, vCurrent, ctx);
        transfer::download(*livingCells, vLiving, ctx);

        for (uint i = 0; i < tripletsBasis.get_count(); i++) {
            PLOG << i
                 << " " << vTBasis[i]
                 << " " << vTFollowers[i]
                 << " [" << vLiving[vTBasis[i]]
                 << ", " << vLiving[vTFollowers[i]]
                 << "] " << vCurrent[vTBasis[i]]
                 << " -> " << vNext[vTBasis[i]]
                 << " " << vCurrent[vTFollowers[i]]
                 << " -> " << vNext[vTFollowers[i]]
                 << std::endl;
        }
    }
    
    delete tripletCurrentState;
    delete tripletNextState;
    delete livingCells;
    LOG << std::endl << "END CellularAutomaton" << std::endl;
}
