#include "TrackletCircleFitter.h"

clever::vector<float, 1>*
TrackletCircleFitter::run(const HitCollection &hits,
                               const TrackletCollection &tracklets,
                               const clever::vector<uint, 1> &validTrackletsIndices,
                               uint nThreads,
                               bool printPROLIX) const
{
    LOG << std::endl << "BEGIN TrackletCircleFitter" << std::endl;
    nThreads = std::min(nThreads, trackletCircleFitterStore.getWorkGroupSize());
    const uint nTracklets = tracklets.size();
    const uint nGroups = uint(std::max(1.0f, ceil(float(nTracklets) / nThreads)));
    clever::vector<float, 1> * const tripletPt  = new clever::vector<float, 1>(nTracklets, ctx);

    cl_event evt;
    evt = trackletCircleFitterStore.run(
            //input
            hits.transfer.buffer(GlobalX()),
            hits.transfer.buffer(GlobalY()),
            hits.transfer.buffer(GlobalZ()),
            tracklets.transfer.buffer(TrackletHit1()),
            tracklets.transfer.buffer(TrackletHit2()),
            tracklets.transfer.buffer(TrackletHit3()),
            validTrackletsIndices.get_mem(),
            //output
            tripletPt->get_mem(),
            //workload
            validTrackletsIndices.get_count(),
            //configuration
            range(nGroups * nThreads),
            range(nThreads));
    TrackletCircleFitter::events.push_back(evt);

    if(((PROLIX) && printPROLIX)){
        PLOG << "Fetching triplet Pt and Eta...";
        std::vector<float> vPt(tripletPt->get_count());
        transfer::download(*tripletPt, vPt, ctx);

        PLOG << "done" << std::endl;
        PLOG << "index, Pt" << std::endl;
        for (uint i = 0; i < tripletPt->get_count(); i++) {
            PLOG << i << " " << vPt[i] << std::endl;
        }
    }

    LOG << std::endl << "END TrackletCircleFitter" << std::endl;
    return tripletPt;
}
