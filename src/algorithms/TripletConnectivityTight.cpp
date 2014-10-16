#include "TripletConnectivityTight.h"


std::tuple<clever::vector<uint, 1>*, clever::vector<uint, 1>*, clever::vector<uint, 1>*, clever::vector<uint, 1>* >
TripletConnectivityTight::run(const HitCollection &hits, TrackletCollection &trackletsInitial,
                              const float dEtaCut, const uint nThreads, bool printPROLIX) const
{
    LOG << std::endl << "BEGIN TripletConnectivityTight" << std::endl;

    uint nTracklets = trackletsInitial.size();
    uint nOracleCount = nTracklets;
    //uint nOracleCount = std::ceil(nTracklets / (float) sizeof(uint));
    uint nGroups = (uint) std::max(1.0f, ceil(((float) nTracklets) / nThreads));

    PLOG << "Initial tracklets: " << nTracklets << std::endl;
    cl_event evt;

    LOG << "Running Triplet Eta Calculator kernel...";

    clever::vector<float, 1> m_tripletEta(nTracklets, ctx);

    evt = tripletEtaCalculatorStore.run(
              //input
              hits.transfer.buffer(GlobalX()),
              hits.transfer.buffer(GlobalY()),
              hits.transfer.buffer(GlobalZ()),
              trackletsInitial.transfer.buffer(TrackletHit1()),
              trackletsInitial.transfer.buffer(TrackletHit3()),
              //output
              m_tripletEta.get_mem(),
              //workload
              nTracklets,
              //thread config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done." << std::endl;

    if (((PROLIX) && printPROLIX)) {
        PLOG << "Fetching oracle...";
        std::vector<float> vTripletEta(m_tripletEta.get_count());
        transfer::download(m_tripletEta, vTripletEta, ctx);
        PLOG << "done" << std::endl;
        PLOG << "Eta:" << std::endl;
        for (auto i : vTripletEta) {
            PLOG << i << std::endl;
        }
        PLOG << std::endl;
    }

    clever::vector<uint, 1> m_oracle(0, nOracleCount, ctx);
    clever::vector<uint, 1> m_trackletFollowerPrefixSum(0, nTracklets + 1, ctx);

    LOG << "Running connectivity tight count kernel...";

    evt = tripletConnectivityTightCount.run(
              //input
              trackletsInitial.transfer.buffer(TrackletHit2()),
              trackletsInitial.transfer.buffer(TrackletHit3()),
              trackletsInitial.transfer.buffer(TrackletHit1()),
              trackletsInitial.transfer.buffer(TrackletHit2()),
              m_tripletEta.get_mem(),
              dEtaCut,
              //output
              m_trackletFollowerPrefixSum.get_mem(),
              m_oracle.get_mem(),
              nTracklets,
              //thread config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done." << std::endl;

    /*
        if (((PROLIX) && printPROLIX)){
            PLOG << "Fetching oracle...";
            std::vector<uint> vOracle(m_oracle.get_count());
            transfer::download(m_oracle, vOracle, ctx);
            PLOG << "done" << std::endl;
            PLOG << "Oracle:" << std::endl;
            for(auto i : vOracle){
                //for (uint j = 0; j < sizeof(uint); j++){
                //    PLOG << (uint)((i & (1 << j)) >> j) << ' ';z
                //}
                PLOG << i << ' ';
            }
            PLOG << std::endl;
        }
    */

    /*
        if (((PROLIX) && printPROLIX)){
            PLOG << "Fetching connectivity count...";
            std::vector<uint> vTrackletFollowerCount(m_trackletFollowerPrefixSum.get_count());
            PLOG << "done." << std::endl;
            transfer::download(m_trackletFollowerPrefixSum, vTrackletFollowerCount, ctx);
            for(auto  i : vTrackletFollowerCount){
                PLOG << i << ' ';
            }
           PLOG << std::endl;
        }
    */

    uint nTrackletConnectablePairs;
    PrefixSum prefixSum(ctx);
    evt = prefixSum.run(
              m_trackletFollowerPrefixSum.get_mem(),
              m_trackletFollowerPrefixSum.get_count(),
              nThreads,
              TripletConnectivityTight::events);
    TripletConnectivityTight::events.push_back(evt);

    transfer::downloadScalar(m_trackletFollowerPrefixSum, nTrackletConnectablePairs, ctx, true,
                             m_trackletFollowerPrefixSum.get_count() - 1, 1, &evt);

    PLOG << "Counted a total of" << nTrackletConnectablePairs
        << " connectable triplet pairs." << std::endl;
    /*
        if (((PROLIX) && printPROLIX)){
            PLOG << "Store Offsets:" << std::endl;
            std::vector<uint> vTrackletFollowerPrefixSum(m_trackletFollowerPrefixSum.get_count());
            transfer::download(m_trackletFollowerPrefixSum, vTrackletFollowerPrefixSum, ctx);
            for(auto  i : vTrackletFollowerPrefixSum){
                PLOG << i << ' ';
            }
            PLOG << std::endl;
        }
    */

    LOG << "Running connectivity tight store kernel, producing " << nTrackletConnectablePairs <<
        " tracklet pairs...";

    clever::vector<uint, 1> * m_connectableTripletBasis = new clever::vector<uint, 1>
        (nTrackletConnectablePairs, ctx);
    
    clever::vector<uint, 1> * m_connectableTripletFollowers = new clever::vector<uint, 1> 
        (nTrackletConnectablePairs, ctx);


    evt = tripletConnectivityTightStore.run(
              //input
              trackletsInitial.transfer.buffer(TrackletHit2()),
              trackletsInitial.transfer.buffer(TrackletHit3()),
              trackletsInitial.transfer.buffer(TrackletHit1()),
              trackletsInitial.transfer.buffer(TrackletHit2()),
              m_tripletEta.get_mem(),
              dEtaCut,
              m_trackletFollowerPrefixSum.get_mem(),
              //output
              m_connectableTripletBasis->get_mem(),
              m_connectableTripletFollowers->get_mem(),
              //workload
              nTracklets,
              //thread config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done." << std::endl;

    if (((PROLIX) && printPROLIX)) {
        PLOG << "Fetching connectable tracklets pairs...";
        std::vector<uint> vConnectableTripletBasis(m_connectableTripletBasis->get_count());
        std::vector<uint> vConnectableTripletFollowers(m_connectableTripletFollowers->get_count());
        transfer::download(*m_connectableTripletBasis, vConnectableTripletBasis, ctx);
        transfer::download(*m_connectableTripletFollowers, vConnectableTripletFollowers, ctx);
        PLOG << "done" << std::endl;
        PLOG << "Tracklet connectable pairs: " << std::endl;
        for (uint i = 0; i < m_connectableTripletFollowers->get_count(); i++) {
            uint hit11 = trackletsInitial.getValue(TrackletHit1(), vConnectableTripletBasis[i]);
            uint hit12 = trackletsInitial.getValue(TrackletHit2(), vConnectableTripletBasis[i]);
            uint hit13 = trackletsInitial.getValue(TrackletHit3(), vConnectableTripletBasis[i]);
            uint hit21 = trackletsInitial.getValue(TrackletHit1(), vConnectableTripletFollowers[i]);
            uint hit22 = trackletsInitial.getValue(TrackletHit2(), vConnectableTripletFollowers[i]);
            uint hit23 = trackletsInitial.getValue(TrackletHit3(), vConnectableTripletFollowers[i]);
            PLOG << "[" << i << "]"
                 << " "
                 << '[' << vConnectableTripletBasis[i] << " ==> " << vConnectableTripletFollowers[i] << ']'
                 << " <==> "
                 << '[' << hit11 << '-' << hit12 << '-' << hit13 << ']'
                 << " ==> "
                 << '[' << hit21 << '-' << hit22 << '-' << hit23 << ']'
                 << std::endl;
        }
        PLOG << std::endl;
    }

    // Fom here, we already have the triplet pairings that share two hits
    // Some triplets cannot aren't connectable already,
    // those that have the their oracle set to 0.
    // There's no point inserting them into the automata
    // moreover, they are tracks already
    //
    // Next step is to filter those that are connectable a more complex critera
    // before, we should have the list of the indexes of the triplets that are connectable
    // so we are going to expand the oracle into a vector of indexes
    //
    // Given a predicate (ie, the oracle) and a prefixSum of the predicate
    // the storage index of a given index that verifies the predicate is given
    // by the prefixSum of its index
    // Store condition:
    // if (pred[i]) store[pSum[i]] = data[i];
    //    i = [0 1 2 3 4 5 6]
    // data = [A B C D E F G]
    // pred = [1 0 1 1 0 0 1]
    // pSum = [0 1 1 2 3 3 3 4] (the last is the length of the filtered stream)
    // stor = [A C D G]; length = 4
    // actually the oracle is not necesary because:
    // (1) pSum[i + 1] == pSum[i] + pred[i]
    // (2) pred[i] is either 0 or 1 => pSum is monotonically increasing
    // hence:
    // (3) pred[i] == 1 <==> pSum[i] != pSum[i + 1]

    //TODO make a class for stream compaction

    // Counting the total amount of connectable triplets

    /*
        LOG << "Initializing oracle prefix sum...";
        clever::vector<uint, 1> m_prefixSum(0, nOracleCount, ctx);
        LOG << "done[" << m_prefixSum.get_count()  << "]" << std::endl;
        LOG << "Running popcount kernel on the oracle...";
        nGroups = (uint) std::max(1.0f, ceil(((float) nOracleCount) / nThreads));
        evt = filterPopCount.run(
                  m_oracle.get_mem(), m_prefixSum.get_mem(), nOracleCount,
                  //threads
                  range(nGroups * nThreads),
                  range(nThreads));

        TripletConnectivityTight::events.push_back(evt);
        LOG << "done" << std::endl;

        if (((PROLIX) && printPROLIX)) {
            PLOG << "Fetching oracle prefix sum...";
            std::vector<uint> cPrefixSum(m_prefixSum.get_count());
            transfer::download(m_prefixSum, cPrefixSum, ctx);
            PLOG << "done" << std::endl;
            PLOG << "Prefix sum: ";
            for (auto i : cPrefixSum) {
                PLOG << i << " ; ";
            }
            PLOG << std::endl;
        }
    */
    LOG << "Inverting connectivity oracle...";
    clever::vector<uint, 1> m_invOracle(m_oracle.get_count(), ctx);
    evt = predicateInversionStore.run(
              //input
              m_oracle.get_mem(),
              //output
              m_invOracle.get_mem(),
              //workload
              m_oracle.get_count() - 1,
              //thread config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done." << std::endl;

    LOG << "Running prefix sum for connectivity " << nTracklets << " tracklets...";
    evt = prefixSum.run(
              m_oracle.get_mem(),
              m_oracle.get_count(),
              nThreads,
              TripletConnectivityTight::events);
    TripletConnectivityTight::events.push_back(evt);

    uint nConnectableTracklets;
    transfer::downloadScalar(m_oracle, nConnectableTracklets, ctx, true, m_oracle.get_count() - 1,
                             1, &evt);
    LOG << "done. Connectable Tracklets: " << nConnectableTracklets << std::endl;

    LOG << "Running prefix sum for non-connectivity " << nTracklets << " tracklets...";

    evt = prefixSum.run(
              m_invOracle.get_mem(),
              m_invOracle.get_count(),
              nThreads,
              TripletConnectivityTight::events);
    TripletConnectivityTight::events.push_back(evt);

    uint nNonConnectableTracklets;
    transfer::downloadScalar(m_invOracle, nNonConnectableTracklets, ctx, true,
                             m_invOracle.get_count() - 1, 1, &evt);

    LOG << "done. Non-connectable Tracklets: " << nNonConnectableTracklets << std::endl;

    if ((PROLIX) && printPROLIX) {
        PLOG << "Fetching oracle prefix sum...";
        std::vector<uint> vOracle(m_oracle.get_count());
        transfer::download(m_oracle, vOracle, ctx);
        PLOG << "done" << std::endl;
        PLOG << "Prefix sum: ";
        for (auto i : vOracle) {
            PLOG << i << " ; ";
        }
        PLOG << std::endl;
    }
    /*
        if (((PROLIX) && printPROLIX)) {
            PLOG << "Fetching inversed oracle prefix sum...";
            std::vector<uint> vInvOracle(m_invOracle.get_count());
            transfer::download(m_invOracle, vInvOracle, ctx);
            PLOG << "done" << std::endl;
            PLOG << "Prefix sum: ";
            for (auto i : vInvOracle) {
                PLOG << i << " ; ";
            }
            PLOG << std::endl;
        }
    */

    //Getting the indices of the connectable triplets
    LOG << "Running stream compaction for connectable tracklets over " << nTracklets <<
        " tracklets...";
    clever::vector<uint, 1> * m_conectableTripletIndexes = new clever::vector<uint, 1>
    (nConnectableTracklets, ctx);
    evt = streamCompactionGetValidIndexesStore.run(
              //input
              m_oracle.get_mem(),
              //output
              m_conectableTripletIndexes->get_mem(),
              //workload
              m_oracle.get_count() - 1,
              //thread config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done. Produced " << m_conectableTripletIndexes->get_count() <<
        " indexes of connectable tracklets..." << std::endl;

    //Getting the indices of the non-connectable triplets

    LOG << "Running stream compaction for non-connectable tracklets over " << nTracklets <<
        " tracklets...";

    //TODO SIZE 0 SHOULD BE A VALID SIZE, FIX CLEVER!!!!
    clever::vector<uint, 1> * m_nonConectableTripletIndexes = NULL;
    if (nNonConnectableTracklets > 0) {
        m_nonConectableTripletIndexes = new clever::vector<uint, 1>(nNonConnectableTracklets, ctx);
        evt = streamCompactionGetValidIndexesStore.run(
                  //input
                  m_invOracle.get_mem(),
                  //output
                  m_nonConectableTripletIndexes->get_mem(),
                  //workload
                  m_invOracle.get_count() - 1,
                  //thread config
                  range(nGroups * nThreads),
                  range(nThreads));
        TripletConnectivityTight::events.push_back(evt);
        LOG << "done. Produced " << m_nonConectableTripletIndexes->get_count()
            << " indexes of non-connectable tracklets..." << std::endl;
    }

    /*
        if(((PROLIX) && printPROLIX)){
            PLOG << "Fetching connectable triplet indexes...";
            std::vector<uint> vConectableTripletIndexes(m_conectableTripletIndexes->get_count());
            transfer::download(*m_conectableTripletIndexes, vConectableTripletIndexes, ctx);
            PLOG << "done" << std::endl;
            PLOG << "Indexes: ";
            for (auto i : vConectableTripletIndexes) {
                PLOG << i << " ; ";
            }
            PLOG << std::endl;
        }
    */
    /*
        if(((PROLIX) && printPROLIX)){
            PLOG << "Fetching non-connectable triplet indexes...";
            std::vector<uint> vNonConectableTripletIndexes(m_nonConectableTripletIndexes->get_count());
            transfer::download(*m_nonConectableTripletIndexes, vNonConectableTripletIndexes, ctx);
            PLOG << "done" << std::endl;
            PLOG << "Indexes: ";
            for (auto i : vNonConectableTripletIndexes) {
                PLOG << i << " ; ";
            }
            PLOG << std::endl;
        }
    */

#ifdef COMPACTION
    LOG << "Running stream filtering for connectable tracklets over " << nTracklets <<
        " tracklets...";
    clever::vector<uint, 1> m_conectableTripletH1(m_conectableTripletIndexes->get_count(), ctx);
    clever::vector<uint, 1> m_conectableTripletH2(m_conectableTripletIndexes->get_count(), ctx);
    clever::vector<uint, 1> m_conectableTripletH3(m_conectableTripletIndexes->get_count(), ctx);

    evt = streamCompactionFilter3StreamsStore.run(
              //input
              trackletsInitial.transfer.buffer(TrackletHit1()),
              trackletsInitial.transfer.buffer(TrackletHit2()),
              trackletsInitial.transfer.buffer(TrackletHit3()),
              m_conectableTripletIndexes->get_mem(),
              //output
              m_conectableTripletH1.get_mem(),
              m_conectableTripletH2.get_mem(),
              m_conectableTripletH3.get_mem(),
              //workload
              m_conectableTripletIndexes->get_count(),
              //config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    LOG << "done. Produced " << m_conectableTripletIndexes->get_count()
        << " tracklets..." << std::endl;

    LOG << "Running stream filtering for non-connectable tracklets over "
        << nTracklets << " tracklets...";

    clever::vector<uint, 1> m_nonConectableTripletH1(m_nonConectableTripletIndexes->get_count(), ctx);
    clever::vector<uint, 1> m_nonConectableTripletH2(m_nonConectableTripletIndexes->get_count(), ctx);
    clever::vector<uint, 1> m_nonConectableTripletH3(m_nonConectableTripletIndexes->get_count(), ctx);

    evt = streamCompactionFilter3StreamsStore.run(
              //input
              trackletsInitial.transfer.buffer(TrackletHit1()),
              trackletsInitial.transfer.buffer(TrackletHit2()),
              trackletsInitial.transfer.buffer(TrackletHit3()),
              m_nonConectableTripletIndexes->get_mem(),
              //output
              m_nonConectableTripletH1.get_mem(),
              m_nonConectableTripletH2.get_mem(),
              m_nonConectableTripletH3.get_mem(),
              //workload
              m_nonConectableTripletIndexes->get_count(),
              //config
              range(nGroups * nThreads),
              range(nThreads));
    TripletConnectivityTight::events.push_back(evt);
    
    LOG << "done. Produced " << m_nonConectableTripletIndexes->get_count()
        << " tracklets..." << std::endl;
    //Now we have indexes of triplets that are connectable and those that are not
    //Merge list of triplets rejected in the first phase and those of the second

    //Return the collection of rejected triplets and the collection of connectable triplets
#endif

    /*
    std::cerr << nTracklets << ' ' << nConnectableTracklets
              << ' ' << nNonConnectableTracklets
              << ' ' << nTrackletConnectablePairs
              << std::endl;
    */
    LOG << std::endl << "END TripletConnectivityTight" << std::endl;

    return std::make_tuple(m_connectableTripletBasis, m_connectableTripletFollowers,
                           m_conectableTripletIndexes, m_nonConectableTripletIndexes);
}
