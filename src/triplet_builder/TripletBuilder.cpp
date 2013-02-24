/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke
 */

#include <iostream>
#include <iomanip>
#include <set>
#include <fcntl.h>

#include <boost/program_options.hpp>

#include <clever/clever.hpp>

#include <datastructures/test/HitCollectionData.h>
#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Dictionary.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>

#include <datastructures/serialize/Event.pb.h>

#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/HitSorterZ.h>
#include <algorithms/HitSorterPhi.h>
#include <algorithms/PrefixSum.h>
#include <algorithms/BoundarySelectionZ.h>
#include <algorithms/BoundarySelectionPhi.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "RuntimeRecord.h"

#include "lib/ccolor.h"
#include "lib/CSV.h"

float getTIP(const Hit & p1, const Hit & p2, const Hit & p3){
	//circle fit
	//map points to parabloid: (x,y) -> (x,y,x^2+y^2)
	float3 pP1 (p1.globalX(),
			p1.globalY(),
			p1.globalX() * p1.globalX() + p1.globalY() * p1.globalY());

	float3 pP2 (p2.globalX(),
			p2.globalY(),
			p2.globalX() * p2.globalX() + p2.globalY() * p2.globalY());

	float3 pP3 (p3.globalX(),
			p3.globalY(),
			p3.globalX() * p3.globalX() + p3.globalY() * p3.globalY());

	//span two vectors
	float3 a(pP2.x - pP1.x, pP2.y - pP1.y, pP2.z - pP1.z);
	float3 b(pP3.x - pP1.x, pP3.y - pP1.y, pP3.z - pP1.z);

	//compute unit cross product
	float3 n(a.y*b.z - a.z*b.y,
			a.z*b.x - a.x*b.z,
			a.x*b.y - a.y*b.x );
	float value = sqrt(n.x*n.x+n.y*n.y+n.z*n.z);
	n.x /= value; n.y /= value; n.z /= value;

	//formula for orign and radius of circle from Strandlie et al.
	float2 cOrigin((-n.x) / (2*n.z),
			(-n.y) / (2*n.z));

	float c = -(n.x*pP1.x + n.y*pP1.y + n.z*pP1.z);

	float cR = sqrt((1 - n.z*n.z - 4 * c * n.z) / (4*n.z*n.z));

	//find point of closest approach to (0,0) = cOrigin + cR * unitVec(toOrigin)
	float2 v(-cOrigin.x, -cOrigin.y);
	value = sqrt(v.x*v.x+v.y*v.y);
	v.x /= value; v.y /= value;;

	float2 pCA = (cOrigin.x + cR*v.x,
			cOrigin.y + cR*v.y);

	//TIP = distance of point of closest approach to origin
	float tip = sqrt(pCA.x*pCA.x + pCA.y*pCA.y);

	return tip;
}

struct tEtaData{
	uint valid;
	uint fake;
	uint missed;

	tEtaData() : valid(0), fake(0), missed(0) {}
};

float getEta(const Hit & innerHit, const Hit & outerHit){
	float3 p (outerHit.globalX() - innerHit.globalX(), outerHit.globalY() - innerHit.globalY(), outerHit.globalZ() - innerHit.globalZ());

	double t(p.z/std::sqrt(p.x*p.x+p.y*p.y));
	return asinh(t);
}

float getEta(const PB_Event::PHit & innerHit, const PB_Event::PHit & outerHit){
	float3 p (outerHit.position().x() - innerHit.position().x(), outerHit.position().y() - innerHit.position().y(), outerHit.position().z() - innerHit.position().z());

	double t(p.z/std::sqrt(p.x*p.x+p.y*p.y));
	return asinh(t);
}

float getEtaBin(float eta){
	float binwidth = 0.1;
	return binwidth*floor(eta/binwidth);
}

RuntimeRecord buildTriplets(std::string filename, uint tracks, float minPt, uint threads, bool verbose = false, bool useCPU = false, int maxEvents = 1) {
	//
	clever::context *contx;

	if(!useCPU){
		try{
			//try gpu
			clever::context_settings settings = clever::context_settings::default_gpu();
			settings.m_profile = true;

			contx = new clever::context(settings);
			std::cout << "Using GPGPU" << std::endl;
		} catch (const std::runtime_error & e){
			//if not use cpu
			clever::context_settings settings = clever::context_settings::default_cpu();
			settings.m_profile = true;

			contx = new clever::context(settings);
			std::cout << "Using CPU" << std::endl;
		}
	} else {
		clever::context_settings settings = clever::context_settings::default_cpu();
		settings.m_profile = true;

		contx = new clever::context(settings);
		std::cout << "Using CPU" << std::endl;
	}

	//load radius dictionary
	Dictionary dict;

	std::ifstream radiusDictFile("radiusDictionary.dat");
	CSVRow row;
	while(radiusDictFile >> row)
	{
		dict.addWithValue(atof(row[1].c_str()));
	}
	radiusDictFile.close();

	//load detectorGeometry
	DetectorGeometry geom;

	std::map<uint, std::pair<float, float> > exRadius; //min, max
	for(int i = 1; i <= 13; ++i){
		exRadius[i]= std::make_pair(10000.0, 0.0);
	}

	std::ifstream detectorGeometryFile("detectorRadius.dat");
	while(detectorGeometryFile >> row)
	{
		uint detId = atoi(row[0].c_str());
		uint layer = atoi(row[2].c_str());
		uint dictEntry = atoi(row[1].c_str());

		geom.addWithValue(detId, layer, dictEntry);
		DictionaryEntry entry(dict, dictEntry);

		if(entry.radius() > exRadius[layer].second) //maxiRadius
			exRadius[layer].second = entry.radius();
		if(entry.radius() < exRadius[layer].first) //minRadius
			exRadius[layer].first = entry.radius();

	}
	detectorGeometryFile.close();

	GeometrySupplement geomSupplement;
	for(auto mr : exRadius){
		geomSupplement.addWithValue(mr.first, mr.second.first, mr.second.second);
	}

	//transfer geometry to device
	geom.transfer.initBuffers(*contx, geom);
	geom.transfer.toDevice(*contx, geom);

	geomSupplement.transfer.initBuffers(*contx, geomSupplement);
	geomSupplement.transfer.toDevice(*contx, geomSupplement);

	dict.transfer.initBuffers(*contx, dict);
	dict.transfer.toDevice(*contx, dict);

	//define statistics variables
	RuntimeRecord result;
	std::map<float, tEtaData> etaHist;
	std::ofstream validTIP("validTIP.csv", std::ios::trunc);
	std::ofstream fakeTIP("fakeTIP.csv", std::ios::trunc);

	//load hit file
	PB_Event::PEventContainer pContainer;

	int fd = open(filename.c_str(), O_RDONLY);
	google::protobuf::io::FileInputStream fStream(fd);
	google::protobuf::io::CodedInputStream cStream(&fStream);

	cStream.SetTotalBytesLimit(536870912, -1);

	if(!pContainer.ParseFromCodedStream(&cStream)){
		std::cerr << "Could not read protocol buffer" << std::endl;
		return result;
	}
	cStream.~CodedInputStream();
	fStream.Close();
	fStream.~FileInputStream();
	close(fd);

	uint lastEvent;
	if(maxEvents == -1)
		lastEvent = pContainer.events_size();
	else
		lastEvent = min(maxEvents, pContainer.events_size());

	for(uint event = 0; event < lastEvent; ++event){

		PB_Event::PEvent pEvent = pContainer.events(event);

		std::cout << "Started processing Event " << pEvent.eventnumber() << " LumiSection " << pEvent.lumisection() << " Run " << pEvent.runnumber() << std::endl;

		//configure hit loader
		const uint maxLayer = 3;
		const uint nSectorsZ = 10;
		const uint nSectorsPhi = 8;
		LayerSupplement layerSupplement(maxLayer);
		Grid grid(maxLayer, nSectorsZ,nSectorsPhi);

		HitCollection hits;
		HitCollection::tTrackList validTracks = hits.addEvent(pEvent, geom, layerSupplement, minPt, tracks, true, maxLayer);

		std::cout << "Loaded " << validTracks.size() << " tracks with minPt " << minPt << " GeV and " << hits.size() << " hits" << std::endl;

		if(verbose){
			for(uint i = 1; i <= maxLayer; ++i)
				std::cout << "Layer " << i << ": " << layerSupplement[i-1].getNHits() << " hits" << "\t Offset: " << layerSupplement[i-1].getOffset() << std::endl;


			//output sector borders
			/*std::cout << "Layer";
		for(int i = 0; i <= nSectorsPhi; ++i){
			std::cout << "\t" << -M_PI + i * (2*M_PI / nSectorsPhi);
		}
		std::cout << std::endl;

		for(int i = 0; i < maxLayer; i++){
			std::cout << i+1;
			for(int j = 0; j <= nSectorsPhi; j++){
				std::cout << "\t" << layerSupplement[i].sectorBorders[j];
			}
			std::cout << std::endl;
		}*/
		}

		/*****


	cl_device_id device;
	ERROR_HANDLER(
			ERROR = clGetCommandQueueInfo(contx->default_queue(), CL_QUEUE_DEVICE, sizeof(cl_device_id), &device, NULL));

	cl_ulong localMemSize;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL));
	cl_ulong maxAlloc;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &maxAlloc, NULL));
	size_t maxParam;
	ERROR_HANDLER(
			ERROR = clGetDeviceInfo(device, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), &maxParam, NULL));

	std::cout << "LocalMemSize: " << localMemSize << std::endl;
	std::cout << "MaxAlloc: " << maxAlloc << std::endl;
	std::cout << "MaxParam: " << maxParam << std::endl;



		 ********/

		//transer everything to gpu
		hits.transfer.initBuffers(*contx, hits);
		hits.transfer.toDevice(*contx, hits);

		//transferring layer supplement
		layerSupplement.transfer.initBuffers(*contx, layerSupplement);
		layerSupplement.transfer.toDevice(*contx,layerSupplement);
		//initializating grid
		grid.transfer.initBuffers(*contx,grid);
		grid.config.upload(*contx);

		/*for(int i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << " Coordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		std::cout << " DetId: " << hit.getValue<DetectorId>() << " DetLayer: " << hit.getValue<DetectorLayer>();
		std::cout << " Event: " << hit.getValue<EventNumber>() << " HitId: " << hit.getValue<HitId>() << std::endl;
	}*/

		cl_ulong runtimeBuildGrid = 0;

		//sort hits on device in Z
		HitSorterZ sorterZ(*contx);
		runtimeBuildGrid += sorterZ.run(hits, threads,maxLayer,layerSupplement);

		//verify sorting
		bool valid = true;
		for(uint l = 1; l <= maxLayer; ++l){
			float lastZ = -9999;
			for(uint i = 0; i < layerSupplement[l-1].getNHits(); ++i){
				Hit hit(hits, layerSupplement[l-1].getOffset() + i);
				if(hit.globalZ() < lastZ){
					std::cerr << "Layer " << l << " : " << lastZ <<  "|" << hit.globalZ() << std::endl;
					valid = false;
				}
				lastZ = hit.globalZ();
			}
		}

		if(!valid)
			std::cerr << "Not sorted properly in Z" << std::endl;
		else
			std::cout << "Sorted correctly in Z" << std::endl;

		BoundarySelectionZ boundSelectZ(*contx);
		runtimeBuildGrid += boundSelectZ.run(hits, threads, maxLayer, layerSupplement, grid);

		//sort hits on device in Phi
		HitSorterPhi sorterPhi(*contx);
		runtimeBuildGrid += sorterPhi.run(hits, threads,maxLayer,layerSupplement, grid);

		//verify sorting
		valid = true;
		for(uint l = 1; l <= maxLayer; ++l){
			LayerGrid layerGrid(grid, l);
			for(uint s = 1; s <= grid.config.nSectorsZ; ++s){

				float lastPhi = - M_PI;
				for(uint h = layerGrid(s-1); h < layerGrid(s); ++h){
					Hit hit(hits, layerSupplement[l-1].getOffset() + h);

					//check phi
					if(hit.phi() < lastPhi){
						std::cerr << "Layer " << l << " : " << lastPhi <<  "|" << hit.phi() << std::endl;
						valid = false;
					}
					lastPhi = hit.phi();

					//check correct z sector
					if(!(grid.config.boundaryValuesZ[s-1] <= hit.globalZ() && hit.globalZ() <= grid.config.boundaryValuesZ[s])){
						std::cerr << "Layer " << l << " : zAct: " << hit.globalZ() <<  " in sector [" << grid.config.boundaryValuesZ[s-1] << ", " << grid.config.boundaryValuesZ[s] << "]" << std::endl;
						valid = false;
					}
				}
			}
		}

		if(!valid)
			std::cerr << "Not sorted properly in Phi" << std::endl;
		else
			std::cout << "Sorted correctly in Phi" << std::endl;



		/*for(int i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << " Coordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		std::cout << " Phi: " << atan2(hit.globalY(), hit.globalX()) << std::endl;
	}*/

		BoundarySelectionPhi boundSelectPhi(*contx);
		runtimeBuildGrid += boundSelectPhi.run(hits, threads, maxLayer, layerSupplement, grid);

		//output grid
		for(uint l = 1; l <= maxLayer; ++l){
			std::cout << "Layer: " << l << std::endl;

			//output z boundaries
			std::cout << "z/phi\t\t";
			for(uint i = 0; i <= grid.config.nSectorsZ; ++i){
				std::cout << grid.config.boundaryValuesZ[i] << "\t";
			}
			std::cout << std::endl;

			LayerGrid layerGrid(grid, l);
			for(uint p = 0; p <= grid.config.nSectorsPhi; ++p){
				std::cout << std::setprecision(3) << grid.config.boundaryValuesPhi[p] << "\t\t";
				for(uint z = 0; z <= grid.config.nSectorsZ; ++z){
					std::cout << layerGrid(z,p) << "\t";
				}
				std::cout << std::endl;
			}
		}


		//output hits
		/*for(uint i = 0; i < hits.size(); ++i){
		Hit hit(hits, i);

		std::cout << "[" << i << "]";
		std::cout << "\tTrack: " << hit.getValue<HitId>();
		std::cout << " \tCoordinates: [" << hit.globalX() << ";" << hit.globalY() << ";" << hit.globalZ() << "]";
		//std::cout << " DetId: " << hit.getValue<DetectorId>() << " DetLayer: " << hit.getValue<DetectorLayer>();
		//std::cout << " Event: " << hit.getValue<EventNumber>() << " HitId: " << hit.getValue<HitId>();
		std::cout << std::endl;
	}*/

		//prefix sum test
		/*std::vector<uint> uints(19,100);
	uints.push_back(0);
	clever::vector<uint,1> dUints(uints, *contx);

	PrefixSum psum(*contx);
	uint res = psum.run(dUints, uints.size(), 4, true);
	transfer::download(dUints, uints, *contx);

	for(uint i = 0; i < uints.size(); ++i){
		std::cout << i << ":" << uints[i] << "\t";
	}

	std::cout << std::endl << "Result: " << res << std::endl;

	return RuntimeRecord();*/

		// configure kernel

		int layers[] = {1,2,3};

		float dThetaCut = 0.05;
		float dThetaWindow = 0.1;
		float dPhiCut = 0.1;
		float dPhiWindow = 0.1;
		int pairSpreadZ = 1;
		float tipCut = 0.75;

		//run it
		PairGeneratorSector pairGen(*contx);
		clever::vector<uint2,1> * pairs = pairGen.run(hits, threads, layers, layerSupplement , grid, pairSpreadZ);

		TripletThetaPhiPredictor predictor(*contx);
		clever::vector<uint2,1> * tripletCandidates = predictor.run(hits, geom, geomSupplement, dict, threads, layers, layerSupplement, grid, dThetaWindow, dPhiWindow, *pairs);

		TripletThetaPhiFilter tripletThetaPhi(*contx);
		TrackletCollection * tracklets = tripletThetaPhi.run(hits, geom, geomSupplement, dict, pairs, tripletCandidates,
				threads, layers, layerSupplement, grid, dThetaCut, dPhiCut, tipCut);

		//evaluate it
		std::set<uint> foundTracks;
		uint fakeTracks = 0;

		std::cout << "Found " << tracklets->size() << " triplets:" << std::endl;
		for(uint i = 0; i < tracklets->size(); ++i){
			Tracklet tracklet(*tracklets, i);

			if(tracklet.isValid(hits)){
				//valid triplet
				foundTracks.insert(tracklet.trackId(hits));

				validTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
				etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].valid++;
				if(verbose){
					std::cout << zkr::cc::fore::green;
					std::cout << "Track " << tracklet.trackId(hits) << " : " << tracklet.hit1() << "-" << tracklet.hit2() << "-" << tracklet.hit3();
					std::cout << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
					std::cout << zkr::cc::console << std::endl;
				}
			}
			else {
				//fake triplet
				++fakeTracks;

				fakeTIP << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3())) << std::endl;
				etaHist[getEtaBin(getEta(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit3())))].fake++;
				if(verbose){
					std::cout << zkr::cc::fore::red;
					std::cout << "Fake: " << tracklet.hit1() << "[" << hits.getValue(HitId(),tracklet.hit1()) << "]";
					std::cout << "-" << tracklet.hit2() << "[" << hits.getValue(HitId(),tracklet.hit2()) << "]";
					std::cout << "-" << tracklet.hit3() << "[" << hits.getValue(HitId(),tracklet.hit3()) << "]";
					std::cout << " TIP: " << getTIP(Hit(hits,tracklet.hit1()), Hit(hits,tracklet.hit2()), Hit(hits,tracklet.hit3()));
					std::cout << zkr::cc::console << std::endl;
				}
			}
		}

		//output not found tracks
		for(auto vTrack : validTracks) {
			if( foundTracks.find(vTrack.first) == foundTracks.end()){
				std::cout << "Didn't find track " << vTrack.first << std::endl;

				PB_Event::PHit innerHit = vTrack.second[0];
				PB_Event::PHit outerHit = vTrack.second[vTrack.second.size()-1];

				etaHist[getEtaBin(getEta(innerHit, outerHit))].missed++;
			}
		}

		std::cout << "Efficiency: " << ((double) foundTracks.size()) / validTracks.size() << " FakeRate: " << ((double) fakeTracks) / tracklets->size() << std::endl;

		RuntimeRecord tmpRes;
		tmpRes.nTracks = foundTracks.size();
		tmpRes.efficiency =  ((double) foundTracks.size()) / validTracks.size();
		tmpRes.fakeRate = ((double) fakeTracks) / tracklets->size();

		//determine runtimes
		tmpRes.buildGrid = runtimeBuildGrid;
		std::cout << "Build Grid: " << runtimeBuildGrid << "ns" << std::endl;

		profile_info pinfo = contx->report_profile(contx->PROFILE_WRITE);
		tmpRes.dataTransferWrite = pinfo.runtime();
		std::cout << "Data Transfer\tWritten: " << pinfo.runtime() << "ns\tRead: ";
		pinfo = contx->report_profile(contx->PROFILE_READ);
		tmpRes.dataTransferRead = pinfo.runtime();
		std::cout << pinfo.runtime() << " ns" << std::endl;


		pinfo = contx->report_profile(PairGeneratorSector::KERNEL_COMPUTE_EVT());
		std::cout << "Pair Generation\tCompute: " << pinfo.runtime() << " ns\tStore: ";
		tmpRes.pairGenComp = pinfo.runtime();
		pinfo = contx->report_profile(PairGeneratorSector::KERNEL_STORE_EVT());
		tmpRes.pairGenStore = pinfo.runtime();
		std::cout << pinfo.runtime() << " ns" << std::endl;


		pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_COMPUTE_EVT());
		tmpRes.tripletPredictComp = pinfo.runtime();
		std::cout << "Triplet Prediction\tCompute: " << pinfo.runtime() << " ns\tStore: ";
		pinfo = contx->report_profile(TripletThetaPhiPredictor::KERNEL_STORE_EVT());
		tmpRes.tripletPredictStore = pinfo.runtime();
		std::cout << pinfo.runtime() << " ns" << std::endl;


		pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_COMPUTE_EVT());
		tmpRes.tripletCheckComp = pinfo.runtime();
		std::cout << "Triplet Checking\tCompute: " << pinfo.runtime() << " ns\tStore: ";
		pinfo = contx->report_profile(TripletThetaPhiFilter::KERNEL_STORE_EVT());
		tmpRes.tripletCheckStore = pinfo.runtime();
		std::cout << pinfo.runtime() << " ns" << std::endl;

		result += tmpRes;

		delete tracklets;
		delete pairs;
		delete tripletCandidates;

	}

	delete contx;

	validTIP.close();
	fakeTIP.close();

	std::ofstream etaData("etaData.csv", std::ios::trunc);

	etaData << "#etaBin, valid, fake, missed" << std::endl;
	for(auto t : etaHist){
		etaData << t.first << "," << t.second.valid << "," << t.second.fake << "," << t.second.missed << std::endl;
	}

	etaData.close();

	return result;
}

int main(int argc, char *argv[]) {

	namespace po = boost::program_options;

	float minPt;
	uint tracks;
	uint threads;
	bool silent;
	bool verbose;
	bool useCPU;
	int maxEvents;
	std::string srcFile;
	std::string configFile;

	po::options_description config("Configuration");
	config.add_options()
		("minPt", po::value<float>(&minPt)->default_value(1.0), "minimum track Pt")
		("tracks", po::value<uint>(&tracks)->default_value(100), "number of valid tracks to load")
		("threads", po::value<uint>(&threads)->default_value(256), "number of threads to use")
		("maxEvents", po::value<int>(&maxEvents)->default_value(1), "number of events to process")
		("src", po::value<std::string>(&srcFile)->default_value("hits_ttbar.reco.pb"), "hit database");

	po::options_description cmdLine("Generic Options");
	cmdLine.add(config);
	cmdLine.add_options()
			("help", "produce help message")
			("silent", po::value<bool>(&silent)->default_value(false)->zero_tokens(), "supress all messages from TripletFinder")
			("verbose", po::value<bool>(&verbose)->default_value(false)->zero_tokens(), "elaborate information")
			("use-cpu", po::value<bool>(&useCPU)->default_value(false)->zero_tokens(), "force using CPU instead of GPU")
			("testSuite", "run entire testSuite")
			("config", po::value<std::string>(&configFile), "config file");

	po::variables_map vm;
	po::store(po::parse_command_line(argc,argv,cmdLine), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << cmdLine << std::endl;
		return 1;
	}

	if(vm.count("config")){
		ifstream ifs(configFile);
		if (!ifs)
		{
			cout << "can not open config file: " << configFile << "\n";
			return 0;
		}
		else
		{
			store(parse_config_file(ifs, config), vm);
			notify(vm);
		}
	}

	if(vm.count("testSuite")){

		uint testCases[] = {1, 10, 50, 100, 200, 300, 500 };

		std::ofstream results("timings.csv", std::ios::trunc);

		results << "#nTracks, dataTransfer, buildGrid, pairGen, tripletPredict, tripletFilter, computation, runtime, efficiency, fakeRate" << std::endl;

		for(uint i : testCases){
			RuntimeRecord res = buildTriplets(srcFile,i,minPt, threads, useCPU, maxEvents);

			results << i << ", " << res.totalDataTransfer() << ", " << res.buildGrid << ", " << res.totalPairGen() << ", " << res.totalTripletPredict() << ", " << res.totalTripletCheck() << ", "  << res.totalComputation() << ", " << res.totalRuntime()
						<< ", " << res.efficiency << ", " << res.fakeRate << std::endl;
		}

		results.close();

		return 0;
	}

	std::streambuf * coutSave = NULL;
	if(silent){
		std::ofstream devNull("/dev/null");
		coutSave = std::cout.rdbuf();
		std::cout.rdbuf(devNull.rdbuf());
	}

	RuntimeRecord res = buildTriplets(srcFile, tracks,minPt, threads, verbose, useCPU, maxEvents);

	if(silent){
		std::cout.rdbuf(coutSave);
	}

	std::cout << "Found: " << res.nTracks << " Tracks with mintPt=" << minPt << " using "
			<< threads << " threads in " << res.totalRuntime() << " ns" << std::endl;
	std::cout << "\tData transfer " << res.totalDataTransfer() << " ns" << std::endl;
	std::cout << "\tBuild grid " << res.buildGrid << " ns" << std::endl;
	std::cout << "\tPairGen "	<< res.totalPairGen() << " ns" << std::endl;
	std::cout << "\tTripletPredict " << res.totalTripletPredict() << " ns" << std::endl;
	std::cout << "\tTripletCheck " << res.totalTripletCheck() << " ns" << std::endl;
	std::cout << "\tTotal Computation "	<< res.totalComputation() << " ns" << std::endl;

}
