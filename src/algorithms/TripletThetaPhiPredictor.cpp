/*
 * TripletThetaPhiPredictor.cpp
 *
 *  Created on: May 13, 2013
 *      Author: dfunke
 */

#include "TripletThetaPhiPredictor.h"

Pairing * TripletThetaPhiPredictor::run(HitCollection & hits, const DetectorGeometry & geom, const GeometrySupplement & geomSupplement, const Dictionary & dict,
			int nThreads, const TripletConfigurations & layerTriplets, const Grid & grid, const Pairing & pairs){

		LOG << "Initializing prefix sum for prediction...";
		clever::vector<uint, 1> m_prefixSum(0, grid.config.nEvents*layerTriplets.size()*nThreads+1, ctx);
		LOG << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		LOG << "Running predict kernel...";
		cl_event evt = predictCount.run(
				//detector geometry
				geom.transfer.buffer(RadiusDict()), dict.transfer.buffer(Radius()), geomSupplement.transfer.buffer(MinRadius()), geomSupplement.transfer.buffer(MaxRadius()),
				grid.transfer.buffer(Boundary()), layerTriplets.transfer.buffer(Layer3()), grid.config.nLayers,
				grid.config.MIN_Z, grid.config.sectorSizeZ(), grid.config.nSectorsZ,
				grid.config.MIN_PHI, grid.config.sectorSizePhi(), grid.config.nSectorsPhi,
				//configuration
				layerTriplets.transfer.buffer(dThetaWindow()), layerTriplets.transfer.buffer(dPhiWindow()),
				// input
				pairs.pairing.get_mem(), pairs.pairingOffsets.get_mem(),
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
				hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()),
				// output
				m_prefixSum.get_mem(),
				local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
				//thread config
				range(nThreads, layerTriplets.size(), grid.config.nEvents),
				range(nThreads, 1,1));
		TripletThetaPhiPredictor::events.push_back(evt);
		LOG << "done" << std::endl;

		if(PROLIX){
			PLOG << "Fetching prefix sum for prediction...";
			std::vector<uint> vPrefixSum(m_prefixSum.get_count());
			transfer::download(m_prefixSum,vPrefixSum,ctx);
			PLOG << "done" << std::endl;
			PLOG << "Prefix sum: ";
			for(auto i : vPrefixSum){
				PLOG << i << " ; ";
			}
			PLOG << std::endl;
		}

		//Calculate prefix sum
		PrefixSum prefixSum(ctx);
		evt = prefixSum.run(m_prefixSum.get_mem(), m_prefixSum.get_count(), nThreads, TripletThetaPhiPredictor::events);
		uint nFoundTripletCandidates;
		transfer::downloadScalar(m_prefixSum, nFoundTripletCandidates, ctx, true, m_prefixSum.get_count()-1, 1, &evt);

		if(PROLIX){
			PLOG << "Fetching prefix sum for prediction...";
			std::vector<uint> vPrefixSum(m_prefixSum.get_count());
			transfer::download(m_prefixSum,vPrefixSum,ctx);
			PLOG << "done" << std::endl;
			PLOG << "Prefix sum: ";
			for(auto i : vPrefixSum){
				PLOG << i << " ; ";
			}
			PLOG << std::endl;
		}

		LOG << "Initializing triplet candidates...";
		Pairing * m_triplets = new Pairing(ctx, nFoundTripletCandidates, grid.config.nEvents, layerTriplets.size());
		LOG << "done[" << m_triplets->pairing.get_count()  << "]" << std::endl;

		LOG << "Running predict store kernel...";
		evt = predictStore.run(
				//geometry
				geom.transfer.buffer(RadiusDict()), dict.transfer.buffer(Radius()), geomSupplement.transfer.buffer(MinRadius()), geomSupplement.transfer.buffer(MaxRadius()),
				grid.transfer.buffer(Boundary()), layerTriplets.transfer.buffer(Layer3()), grid.config.nLayers,
				grid.config.MIN_Z, grid.config.sectorSizeZ(), grid.config.nSectorsZ,
				grid.config.MIN_PHI, grid.config.sectorSizePhi(), grid.config.nSectorsPhi,
				//configuration
				layerTriplets.transfer.buffer(dThetaWindow()), layerTriplets.transfer.buffer(dPhiWindow()),
				//input
				pairs.pairing.get_mem(), pairs.pairingOffsets.get_mem(),
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()),
				hits.transfer.buffer(DetectorId()),
				//oracle
				m_prefixSum.get_mem(),
				// output
				m_triplets->pairing.get_mem(), m_triplets->pairingOffsets.get_mem(),
				//local
				local_param(sizeof(cl_uint), (grid.config.nSectorsZ+1)*(grid.config.nSectorsPhi+1)),
				//thread config
				range(nThreads, layerTriplets.size(), grid.config.nEvents),
				range(nThreads, 1,1));
		TripletThetaPhiPredictor::events.push_back(evt);
		LOG << "done" << std::endl;

		if(PROLIX){
			PLOG << "Fetching triplet candidates...";
			std::vector<uint2> cands = m_triplets->getPairings();
			PLOG <<"done[" << cands.size() << "]" << std::endl;

			PLOG << "Candidates:" << std::endl;
			for(uint i = 0; i < nFoundTripletCandidates; ++i){
				PLOG << "[" << i << "] "  << cands[i].x << "-" << cands[i].y << std::endl;
			}

			PLOG << "Fetching candidates offets...";
			std::vector<uint> candOffsets = m_triplets->getPairingOffsets();
			PLOG <<"done[" << candOffsets.size() << "]" << std::endl;

			PLOG << "Candidate Offsets:" << std::endl;
			for(uint i = 0; i < candOffsets.size(); ++i){
				PLOG << "[" << i << "] "  << candOffsets[i] << std::endl;
			}
		}

		return m_triplets;
	}

