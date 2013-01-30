#pragma once

#include <boost/noncopyable.hpp>

#include <clever/clever.hpp>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/LayerSupplement.h>

#include <algorithms/PrefixSum.h>

using namespace clever;
using namespace std;

/*
	Class which implements to most dump tracklet production possible - combine
	every hit with every other hit.
	The man intention is to test the data transfer and the data structure
 */
class TripletThetaPhiPredictor: private boost::noncopyable
{
private:

	clever::context & ctx;

public:

	TripletThetaPhiPredictor(clever::context & ctext) :
		ctx(ctext),
		tripletThetaPhiPredict(ctext),
		tripletThetaPhiPredictStore(ctext)
{
		// create the buffers this algorithm will need to run
#define DEBUG_OUT
#ifdef DEBUG_OUT
		std::cout << "PredictKernel WorkGroupSize: " << tripletThetaPhiPredict.getWorkGroupSize() << std::endl;
		std::cout << "StoreKernel WorkGroupSize: " << tripletThetaPhiPredictStore.getWorkGroupSize() << std::endl;
#endif
#undef DEBUG_OUT
}

	static std::string KERNEL_COMPUTE_EVT() {return "TripletThetaPhiPredict_COMPUTE";}
	static std::string KERNEL_STORE_EVT() {return "TripletThetaPhiPredict_STORE";}

	clever::vector<uint2,1> * run(HitCollection & hits, const DetectorGeometry & geom, const Dictionary & dict,
			int nThreads, int layers[], const LayerSupplement & layerSupplement,
			float dThetaWindow, float dPhiWindow, const clever::vector<uint2,1> & pairs)
	{

		int nLayer1 = layerSupplement[layers[0]-1].getNHits();
		int nLayer2 = layerSupplement[layers[1]-1].getNHits();
		int nLayer3 = layerSupplement[layers[2]-1].getNHits();

		int nMaxTriplets = pairs.get_count() * nLayer3;

		std::cout << "Initializing oracle for prediction...";
		clever::vector<uint, 1> m_oracle(0, std::ceil(nMaxTriplets / 32.0), ctx);
		std::cout << "done[" << m_oracle.get_count()  << "]" << std::endl;

		std::cout << "Initializing prefix sum for prediction...";
		clever::vector<uint, 1> m_prefixSum(0, nThreads+1, ctx);
		std::cout << "done[" << m_prefixSum.get_count()  << "]" << std::endl;

		std::cout << "Running predict kernel...";
		cl_event evt = tripletThetaPhiPredict.run(
				//detector geometry
				geom.transfer.buffer(RadiusDict()), dict.transfer.buffer(Radius()),
				//configuration
				dThetaWindow, dPhiWindow,
				pairs.get_count(),
				// input
				pairs.get_mem(), (nLayer1+nLayer2), nLayer3,
				hits.transfer.buffer(GlobalX()), hits.transfer.buffer(GlobalY()), hits.transfer.buffer(GlobalZ()), hits.transfer.buffer(DetectorId()), hits.transfer.buffer(HitId()),
				// output
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_COMPUTE_EVT());

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum for prediction...";
		std::vector<uint> prefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

#ifdef DEBUG_OUT
		std::cout << "Fetching oracle for prediction...";
		std::vector<uint> oracle(m_oracle.get_count());
		transfer::download(m_oracle,oracle,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Oracle: ";
		for(auto i : oracle){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

		//Calculate prefix sum
		PrefixSum prefixSum(ctx);
		int nCandidateTriplets = prefixSum.run(m_prefixSum, nThreads, true);

#ifdef DEBUG_OUT
		std::cout << "Fetching prefix sum for prediction...";
		std::vector<uint> prefixSum(m_prefixSum.get_count());
		transfer::download(m_prefixSum,prefixSum,ctx);
		std::cout << "done" << std::endl;
		std::cout << "Prefix sum: ";
		for(auto i : prefixSum){
			std::cout << i << " ; ";
		}
		std::cout << std::endl;
#endif

		std::cout << "Initializing triplet candidates...";
		clever::vector<uint2, 1> * m_triplets = new clever::vector<uint2, 1>(ctx, nCandidateTriplets);
		std::cout << "done[" << m_triplets->get_count()  << "]" << std::endl;

		std::cout << "Running predict store kernel...";
		evt = tripletThetaPhiPredictStore.run(
				//configuration
				pairs.get_count(), layerSupplement[layers[2]-1].getOffset(), nLayer3,
				//input
				pairs.get_mem(),
				m_oracle.get_mem(), m_prefixSum.get_mem(),
				//output
				// output
				m_triplets->get_mem(),
				//thread config
				nThreads);
		std::cout << "done" << std::endl;

		ctx.add_profile_event(evt, KERNEL_STORE_EVT());

		std::cout << "Fetching triplet candidates...";
		std::vector<uint2> triplets(nCandidateTriplets);
		transfer::download(*m_triplets, triplets, ctx);
		std::cout <<"done[" << triplets.size() << "]" << std::endl;

#ifdef DEBUG_OUT
		std::cout << "Triplet Candidates:" << std::endl;
		for(uint2 i : triplets){
			std::cout << i.x << "-" << i.y << std::endl;
		}
#endif

		return m_triplets;
	}

	KERNEL15_CLASS( tripletThetaPhiPredict, cl_mem, cl_mem, double, double, uint,  cl_mem, uint, uint, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem, cl_mem,
			__kernel void tripletThetaPhiPredict(
					//detector geometry
					__global const uchar * detRadius, __global const float * radiusDict,
					//configuration
					double dThetaWindow, double dPhiWindow, uint nPairs,
					// hit input
					__global const uint2 * pairs, uint offset, uint nThirdHits,
					__global const float * hitGlobalX, __global const float * hitGlobalY, __global const float * hitGlobalZ,
					__global const uint * detId, __global const int * hitId,
					// intermeditate data: oracle for hit pair + candidate combination, prefix sum for found tracklets
					__global uint * oracle, __global uint * prefixSum )
	{

		const size_t gid = get_global_id( 0 );
		const size_t lid = get_local_id( 0 );
		const size_t threads = get_global_size( 0 );

		uint workload = nPairs / threads + 1;
		uint i = gid * workload;
		uint end = min(i + workload, nPairs); // for last thread, if not a full workload is present
		uint nFound = 0;

		//printf("id %lu threads %lu workload %i start %i end %i maxEnd %i \n", id, threads, workload, i, end, nPairs);

		for(; i < end; ++i){ //workload loop

			uint firstHit = pairs[i].x;
			uint secondHit = pairs[i].y;

			//theta
			float signRadius = sign(hitGlobalY[secondHit]);
			float theta = atan2( signRadius * sqrt((hitGlobalX[secondHit] - hitGlobalX[firstHit])*(hitGlobalX[secondHit] - hitGlobalX[firstHit])
					+ (hitGlobalY[secondHit] - hitGlobalY[firstHit])*(hitGlobalY[secondHit] - hitGlobalY[firstHit]))
																		, ( hitGlobalZ[secondHit] - hitGlobalZ[firstHit] ));
			float cotThetaLow = tan(M_PI_2_F - (1-dThetaWindow) * theta);
			//int thetaLowSgn = 1 - (fabs(thetaLow) > M_PI_2_F) * 2;
			//thetaLow = (fabs(thetaLow) <= M_PI_2_F) * thetaLow + (fabs(thetaLow) > M_PI_2_F) * (sign(thetaLow)*M_PI_F - thetaLow);

			float cotThetaHigh = tan(M_PI_2_F -(1+dThetaWindow) * theta);
			//int thetaHighSgn = 1 - (fabs(thetaHigh) > M_PI_2_F) * 2;
			//thetaHigh = (fabs(thetaHigh) <= M_PI_2_F) * thetaHigh + (fabs(thetaHigh) > M_PI_2_F) * (sign(thetaHigh)*M_PI_F - thetaHigh);

			//phi
			float phi = atan2((hitGlobalY[secondHit] - hitGlobalY[firstHit]) , ( hitGlobalX[secondHit] - hitGlobalX[firstHit] ));

			float tmp = (1-dPhiWindow) * phi;
			float phiHigh = (1+dPhiWindow) * phi;
			float phiLow = (tmp < phiHigh) * tmp + (tmp > phiHigh) * phiHigh;
			phiHigh = (tmp < phiHigh) * phiHigh + (tmp > phiHigh) * tmp;

			//radius
			float r = signRadius * radiusDict[detRadius[detId[secondHit]]];

			//loop over all third hits
			//TODO[gpu] store hits in more suitable data structure, with phi pre-calculated and (z,phi) sorted
			for(uint j = 0; j < nThirdHits; ++j){
				// check z range
				uint index = offset+j;

				float dr = (signRadius * radiusDict[detRadius[detId[index]]]) - r;

				//z_3 = z_2 + dr * cot(theta) => cot(theta) = tan(pi/2 - theta)
				float tmp = hitGlobalZ[secondHit] + dr * cotThetaLow;
				float zHigh = hitGlobalZ[secondHit] + dr * cotThetaHigh;

				float zLow = (tmp < zHigh) * tmp + (tmp > zHigh) * zHigh;
				zHigh = (tmp < zHigh) * zHigh + (tmp > zHigh) * tmp;

				bool valid = zLow <= hitGlobalZ[index] && hitGlobalZ[index] <= zHigh;

#ifdef DEVICE_CPU
				if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[offset+j]){
					printf("%i-%i-%i [%i]: z exp[%f]: %f - %f; z act: %f\n", firstHit, secondHit, offset+j, hitId[firstHit], hitGlobalZ[secondHit], zLow, zHigh, hitGlobalZ[offset+j]);
					float thetaAct = atan2(sign(hitGlobalY[offset+j])*sqrt((hitGlobalX[offset+j] - hitGlobalX[secondHit])*(hitGlobalX[offset+j] - hitGlobalX[secondHit])
							+ (hitGlobalY[offset+j] - hitGlobalY[secondHit])*(hitGlobalY[offset+j] - hitGlobalY[secondHit]))
							, ( hitGlobalZ[offset+j] - hitGlobalZ[secondHit] ));
					//if(!(thetaLow <= thetaAct && thetaAct <= thetaHigh))
					printf("\ttheta exp[%f]: %f - %f; theta act: %f\n", theta, atan(1/cotThetaLow), atan(1/cotThetaHigh), thetaAct);
					//else {
					float r2 = sqrt(hitGlobalX[secondHit]*hitGlobalX[secondHit] + hitGlobalY[secondHit]*hitGlobalY[secondHit]);
					float r3 = sqrt(hitGlobalX[offset+j]*hitGlobalX[offset+j] + hitGlobalY[offset+j]*hitGlobalY[offset+j]);

					printf("\tdr exp: %f; dr act: %f\n", dr, r3-r2);
					//}
				}
#endif

				// check phi range
				float actPhi = atan2(hitGlobalY[index],hitGlobalX[index]);
				valid = valid * (phiLow <= actPhi && actPhi <= phiHigh);

#ifdef DEVICE_CPU
				if(!valid && hitId[firstHit] == hitId[secondHit] && hitId[secondHit] == hitId[offset+j]){
					printf("%i-%i-%i [%i]: phi exp: %f - %f; phi act: %f\n", firstHit, secondHit, offset+j, hitId[firstHit], phiLow, phiHigh, actPhi);
					//}
				}
#endif

				//if valid update nFound
				nFound = nFound + valid;

				//update oracle
				index = i*nThirdHits + j;
				atomic_or(&oracle[index / 32], (valid << (index % 32)));

				//if(valid)
					//printf("[ %lu ] Found valid candidate %i (%i-%i-%i). Word %i Bit %i\n", id, index, firstHit, secondHit, offset+j, index / 32, index % 32);
			} // end hit loop

		} //end workload loop

		prefixSum[gid] = nFound;
	});

	KERNEL7_CLASS( tripletThetaPhiPredictStore, uint, uint, uint, cl_mem, cl_mem, cl_mem, cl_mem,
				__kernel void tripletThetaPhiPredictStore(
						//configuration
						uint nPairs, uint offset, uint nThirdHits,
						// hit input
						__global const uint2 * pairs,
						// input for oracle and prefix sum
						__global const uint * oracle, __global const uint * prefixSum,
						// output triplet candidates
						__global uint2 * triplets)
		{
			size_t id = get_global_id( 0 );
			size_t threads = get_global_size( 0 );

			uint workload = nPairs / threads + 1;
			uint i = id * workload;
			uint end = min(i + workload, nPairs); // for last thread, if not a full workload is present

			uint pos = prefixSum[id];

			for(; i < end; ++i){

				for(uint j = 0; j < nThirdHits && pos < prefixSum[id+1]; ++j){ // pos < prefixSum[id+1] can lead to thread divergence

					//is this a valid triplet?
					uint index = i*nThirdHits+j;
					bool valid = oracle[index / 32] & (1 << (index % 32));

					//last triplet written on [pos] is valid one
					index = offset + j;
					triplets[pos].x = valid * i;
					triplets[pos].y = valid * index;

					//if(valid)
					//	printf("[ %lu ] Written at %i: %i-%i-%i\n", id, pos, trackletHitId1[pos],trackletHitId2[pos],trackletHitId3[pos]);

					//advance pos if valid
					pos = pos + valid;
				}
			}
		});

};
