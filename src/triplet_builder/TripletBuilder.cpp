/*
 * TripletBuilder.cpp
 *
 *  Created on: Dec 12, 2012
 *      Author: dfunke, Juan Antonio Aldea Armenteros
 */

#include <iostream>
#include <iomanip>
#include <set>
#include <fcntl.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <clever/clever.hpp>

#include <datastructures/HitCollection.h>
#include <datastructures/TrackletCollection.h>
#include <datastructures/DetectorGeometry.h>
#include <datastructures/GeometrySupplement.h>
#include <datastructures/Dictionary.h>
#include <datastructures/EventSupplement.h>
#include <datastructures/LayerSupplement.h>
#include <datastructures/Grid.h>
#include <datastructures/TripletConfiguration.h>
#include <datastructures/Pairings.h>
#include <datastructures/Logger.h>

#include <datastructures/serialize/Event.pb.h>

//#include <algorithms/PairGeneratorSector.h>
#include <algorithms/PairGeneratorBeamspot.h>
#include <algorithms/TripletThetaPhiPredictor.h>
#include <algorithms/TripletThetaPhiFilter.h>
#include <algorithms/GridBuilder.h>
 #include <algorithms/MatrixMul.h>

#include <algorithms/TripletConnectivityTight.h>
#include <algorithms/TrackletCircleFitter.h>
#include <algorithms/CellularAutomaton.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "RuntimeRecord.h"
#include "PhysicsRecord.h"
#include "Parameters.h"
#include "EventLoader.h"

#include "lib/ccolor.h"
#include "lib/CSV.h"

#include "fastfitters/GlobalPoint.h"
#include "fastfitters/FastHelix.h"

std::string g_traxDir;

void printTracks(const std::vector<uint> &vTrackCollection,
                              const std::vector<uint> &vPSum,
                              const TrackletCollection * const tracklets,
                              const HitCollection &hits,
                              const std::vector<float> &vPt,
                              uint totalTrack,
                              float dEtaCut);
clever::context * createContext(ExecutionParameters exec)
{
    LLOG << "Creating context for " << (exec.useCPU ? "CPU" : "GPGPU") << "...";

#define DEBUG_OCL 0
/*
#define DEBUG

#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL  (1<<31)
#if defined(DEBUG) && defined(CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL)
#undef DEBUG_OCL
#define DEBUG_OCL CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL
#endif
*/
//    opencl::PrintInformation();
    clever::context_settings settings;
    if (!exec.useCPU) {
        try {
            //try gpu
            if (exec.computingPlatform == "amd"){
                settings = clever::context_settings::amd_gpu();
            }else if (exec.computingPlatform == "nvidia"){
                settings = clever::context_settings::nvidia_gpu();
            }else{
                settings = clever::context_settings::default_gpu();
            }

            settings.m_profile = true;

            LLOG << "success" << std::endl;
        } catch (const std::runtime_error & e) {
            //if not use cpu
            if (exec.computingPlatform == "amd"){
                settings = clever::context_settings::amd_cpu();
            }else if (exec.computingPlatform == "intel"){
                settings = clever::context_settings::intel_cpu();
            }else{
                settings = clever::context_settings::default_cpu();
            }

            settings.m_profile = true;
            settings.m_cmd_queue_properties |= DEBUG_OCL;

            LLOG << "error: fallback on CPU" << std::endl;
        }
    } else {
        if (exec.computingPlatform == "amd"){
            settings = clever::context_settings::amd_cpu();
        }else if (exec.computingPlatform == "intel"){
            char no_vectorize []= {"CL_CONFIG_USE_VECTORIZER=False"};
            putenv(no_vectorize);
            settings = clever::context_settings::intel_cpu();
        }else{
            settings = clever::context_settings::default_cpu();
        }

        settings.m_profile = true;
        settings.m_cmd_queue_properties |= DEBUG_OCL;

        LLOG << "success" << std::endl;
    }

    return new clever::context(settings);;
}

std::pair<RuntimeRecords, PhysicsRecords> buildTriplets(ExecutionParameters exec,
        EventDataLoadingParameters loader, GridConfig gridConfig, clever::context * contx, float dEtaCut)
{

    //Multiplication mul(*contx);
    //mul.run(9999999,  exec.threads);
    RuntimeRecords runtimeRecords;
    PhysicsRecords physicsRecords;
    std::vector<std::vector<tKernelEvent>> runs;

    {
        //block to ensure proper destruction

        //load radius dictionary
        Dictionary dict;

        std::ifstream radiusDictFile(g_traxDir + "/configs/radiusDictionary.dat");
        CSVRow row;
        while (radiusDictFile >> row) {
            dict.addWithValue(atof(row[1].c_str()));
        }
        radiusDictFile.close();

        //load detectorGeometry
        DetectorGeometry geom;

        std::map<uint, std::pair<float, float> > exRadius; //min, max
        for (int i = 1; i <= 13; ++i) {
            exRadius[i] = std::make_pair(10000.0, 0.0);
        }

        std::ifstream detectorGeometryFile(g_traxDir + "/configs/detectorRadius.dat");
        while (detectorGeometryFile >> row) {
            uint detId = atoi(row[0].c_str());
            uint layer = atoi(row[2].c_str());
            uint dictEntry = atoi(row[1].c_str());

            geom.addWithValue(detId, layer, dictEntry);
            DictionaryEntry entry(dict, dictEntry);

            if (entry.radius() > exRadius[layer].second) { //maxiRadius
                exRadius[layer].second = entry.radius();
            }
            if (entry.radius() < exRadius[layer].first) { //minRadius
                exRadius[layer].first = entry.radius();
            }

        }
        detectorGeometryFile.close();

        GeometrySupplement geomSupplement;
        for (auto mr : exRadius) {
            geomSupplement.addWithValue(mr.first, mr.second.first, mr.second.second);
        }

        //transfer geometry to device
        geom.transfer.initBuffers(*contx, geom);
        geom.transfer.toDevice(*contx, geom);

        geomSupplement.transfer.initBuffers(*contx, geomSupplement);
        geomSupplement.transfer.toDevice(*contx, geomSupplement);

        dict.transfer.initBuffers(*contx, dict);
        dict.transfer.toDevice(*contx, dict);

        //Event Data Loader
        unique_ptr<EventLoader> edLoader(EventLoaderFactory::create(loader.eventLoader, loader));

        uint lastEvent;
        if (loader.maxEvents == -1) {
            lastEvent = edLoader->nEvents();
        } else {
            lastEvent = min((int)(loader.skipEvents + loader.maxEvents), edLoader->nEvents());
        }

        TripletConfigurations layerConfig(loader.minPt);
        loader.maxLayer = layerConfig.loadTripletConfigurationFromFile(g_traxDir + "/configs/" +
                          exec.layerTripletConfigFile, exec.layerTriplets);

        layerConfig.transfer.initBuffers(*contx, layerConfig);
        layerConfig.transfer.toDevice(*contx, layerConfig);

        LOG << "Loaded " << layerConfig.size() << " layer triplet configurations" << std::endl;
        std::cout << "Loaded " << layerConfig.size() << " layer triplet configurations" << std::endl;
        if (VERBOSE) {
            for (uint i = 0; i < layerConfig.size(); ++ i) {
                TripletConfiguration config(layerConfig, i);
                VLOG << "Layers " << config.layer1() << "-" << config.layer2()
                     << "-" << config.layer3() << std::endl;
                VLOG << "\t" << "dThetaCut: " << config.getValue<dThetaCut>() << " sigmaZ: "
                     << config.getValue<sigmaZ>() << std::endl;
                VLOG << "\t" << "dPhiCut: " << config.getValue<dPhiCut>() << " sigmaPhi: "
                     << config.getValue<sigmaPhi>() << std::endl;
                VLOG << "\t" << "tipCut: " << config.getValue<tipCut>() << std::endl;
                VLOG << "\t" << "pairSpreadZ: " << config.getValue<pairSpreadZ>() << " pairSpreadPhi: "
                     << config.getValue<pairSpreadPhi>() << std::endl;
            }
        }

        //configure hit loading
        //number of groups
        const uint evtGroups = (uint) max(1.0f,
                                          ceil(((float) lastEvent - loader.skipEvents) / exec.eventGrouping));

        uint event = loader.skipEvents;
        for (uint eventGroup = 0; eventGroup < evtGroups; ++eventGroup) {
            LOG << "Processing event group: " << eventGroup << std::endl;
            //if last group is not a full group
            uint evtGroupSize = std::min(exec.eventGrouping, lastEvent - event);

            //initialize datastructures
            EventSupplement *eventSupplement = new EventSupplement(evtGroupSize);
            LayerSupplement *layerSupplement = new LayerSupplement(loader.maxLayer, evtGroupSize);

            gridConfig.nLayers = loader.maxLayer;
            gridConfig.nEvents = evtGroupSize;
            Grid grid(gridConfig);

            HitCollection hits;
            //first: uint = evt in group second: tracklist
            std::map<uint, HitCollection::tTrackList> validTracks;
            uint totalValidTracks = 0;
            //delete edLoader;

            uint iEvt = 0;
            do {
                PB_Event::PEvent pEvent = edLoader->getEvent();

                LOG << "Started processing Event " << pEvent.eventnumber() << " LumiSection "
                    << pEvent.lumisection() << " Run " << pEvent.runnumber() << std::endl;
                validTracks[iEvt] = hits.addEvent(pEvent, geom, *eventSupplement, iEvt, *layerSupplement,
                                                  layerConfig, loader.maxTracks, loader.onlyTracks);

                totalValidTracks += validTracks[iEvt].size();
                LOG << "Loaded " << validTracks[iEvt].size()
                    << " tracks with minPt " << loader.minPt
                    << " GeV and " << (*eventSupplement)[iEvt].getNHits()
                    << " hits" << std::endl;
               /*
                 std::cout << "Loaded " << validTracks[iEvt].size()
                    << " tracks with minPt " << loader.minPt
                    << " GeV and " << (*eventSupplement)[iEvt].getNHits()
                    << " hits" << std::endl;
                */
                if (VERBOSE) {
                    for (uint i = 1; i <= loader.maxLayer; ++i) {
                        VLOG << "Layer " << i << ": " << (*layerSupplement)[iEvt * loader.maxLayer + i - 1].getNHits()
                             << " hits" << "\t Offset: " << (*layerSupplement)[iEvt * loader.maxLayer + i - 1].getOffset()
                             << std::endl;
                    }
                }

                ++event;
                ++iEvt;
            } while (iEvt < evtGroupSize && event < lastEvent);


            std::cout << "Loaded " << hits.size() << " hits in " << evtGroupSize << " events" << std::endl;
            if (hits.size() == 0){
                //delete edLoader;
                continue;
            }
            RuntimeRecord runtime(grid.config.nEvents, grid.config.nLayers, layerConfig.size(),
                                  hits.size(), totalValidTracks, exec.threads);

            //transer hits to gpu
            hits.transfer.initBuffers(*contx, hits);
            hits.transfer.toDevice(*contx, hits);

            //transferring layer supplement
            eventSupplement->transfer.initBuffers(*contx, *eventSupplement);
            eventSupplement->transfer.toDevice(*contx, *eventSupplement);

            //transferring layer supplement
            layerSupplement->transfer.initBuffers(*contx, *layerSupplement);
            layerSupplement->transfer.toDevice(*contx, *layerSupplement);

            //initializating grid
            grid.transfer.initBuffers(*contx, grid);
            grid.transfer.toDevice(*contx, grid);

            GridBuilder gridBuilder(*contx);

            //runtime.buildGrid.startWalltime();
            gridBuilder.run(hits, exec.threads, *eventSupplement, *layerSupplement, grid);
            //runtime.buildGrid.stopWalltime();

            /****************************/

/*
            // print updated hit information
            PLOG << "UPDATED HIT IDS" << std::endl;
            for (uint i = 0; i < hits.size(); i++){
                PLOG << "[" << i << "]"
                    << "ID: " << hits.getValue(HitId(), i)
                    << " Layer: " << hits.getValue(DetectorLayer(), i)
                     << " ["  << hits.getValue(GlobalX(), i)
                     << ", " << hits.getValue(GlobalY(), i)
                     << ", " << hits.getValue(GlobalZ(), i)
                     << "]" << std::endl;
            }
*/
            /****************************/

            //run it

            PairGeneratorBeamspot pairGen(*contx);

            //runtime.pairGen.startWalltime();
            Pairing * pairs = pairGen.run(hits, geomSupplement, exec.threads, layerConfig, grid, false);
            //runtime.pairGen.stopWalltime();

            TripletThetaPhiPredictor predictor(*contx);

            //runtime.tripletPredict.startWalltime();
            Pairing * tripletCandidates = predictor.run(hits, geom, geomSupplement, dict, exec.threads,
                                          layerConfig, grid, *pairs, false);
            //runtime.tripletPredict.stopWalltime();

            TripletThetaPhiFilter tripletThetaPhi(*contx);

            //runtime.tripletFilter.startWalltime();
            TrackletCollection * tracklets = tripletThetaPhi.run(hits, grid, *pairs, *tripletCandidates,
                                             exec.threads, layerConfig, false);
            //runtime.tripletFilter.stopWalltime();

            delete pairs;
            delete tripletCandidates;
            contx->clearPerfCounters();
            /*******************************************/
#define MIO
#ifdef MIO
            /*
            std::map<uint, std::map<uint, std::vector<uint>>> tracks;
            for (uint i = 0; i < hits.size(); i++){
                tracks[hits.getValue(EventNumber(), i)][hits.getValue(HitId(), i)].push_back(i);
            }

            for (auto it=tracks.begin(); it!=tracks.end(); ++it){
                auto eventTracks = it->second;
                for (auto it2=eventTracks.begin(); it2!=eventTracks.end(); ++it2){
                    auto track = it2->second;
                    std::sort(track.begin(), track.end());
                }
            }
            */
            /*
            int trackNumber = 1;
            for (auto it=tracks.begin(); it!=tracks.end(); ++it) {
                auto eventTracks = it->second;
                int trackNumberEvent = 1;
                for (auto it2=eventTracks.begin(); it2!=eventTracks.end(); ++it2){
                    std::cout << "Event: " <<  it-> first << " #" << trackNumber << ", " << trackNumberEvent << " SIMULATED " << it->first << " => ";
                    trackNumberEvent++;
                    trackNumber++;
                    auto track = it2->second;
                    for(auto it3 = track.begin(); it3 != track.end(); it3++){
                        std::cout << *it3 << '-';
                    }
                    std::cout << std::endl;
                }
            }
            */

            // On average a dEta = 0.0256 cut reduces the amount of triplets pairs by a 33.7% (stdev 19.6)
            TripletConnectivityTight tripletConnectivityTight(*contx);

            //float dEtaCut  = 0.0256;
            std::cout << "dEtaCut " << dEtaCut << std::endl;
            runtime.tripletConnectivity.startWalltime();

            clever::vector<uint, 1> * m_connectableTripletBasis;
            clever::vector<uint, 1> * m_connectableTripletFollowers;
            clever::vector<uint, 1> * m_conectableTripletIndexes;
            clever::vector<uint, 1> * m_nonConectableTripletIndexes;

            std::tie(m_connectableTripletBasis,
                     m_connectableTripletFollowers,
                     m_conectableTripletIndexes,
                     m_nonConectableTripletIndexes) = tripletConnectivityTight.run(hits, *tracklets, layerConfig, dEtaCut,
                                                   exec.threads, false);
            runtime.tripletConnectivity.stopWalltime();

            if (m_connectableTripletBasis == nullptr){
                std::cout << "No connectable tracklets found, skipping event..." << std::endl;
                LOG << "No connectable tracklets found, skipping event..." << std::endl;
                delete m_connectableTripletBasis;
                delete m_connectableTripletFollowers;
                delete m_conectableTripletIndexes;
                delete m_nonConectableTripletIndexes;
                continue;
            }

            //tripletConnectivityTight.run(hits, *tracklets, dEtaCut, exec.threads, true);
            //std::cerr << tracklets->size() << std::endl;

            // TODO filter the stream here maybe?, we dont want gaps neither for the fitting nor for the CA.

            runtime.trackletCircleFitter.startWalltime();

//#define USE_FAST_FITTERS
#ifdef USE_FAST_FITTERS

            std::vector<float>vPT(tracklets->size());
            for (uint i=0; i< tracklets->size(); i++){
                uint hit1 = tracklets->getValue(TrackletHit1(), i);
                uint hit2 = tracklets->getValue(TrackletHit2(), i);
                uint hit3 = tracklets->getValue(TrackletHit3(), i);
                GlobalPoint vertex(hits.getValue(GlobalX(), hit1), hits.getValue(GlobalY(), hit1), hits.getValue(GlobalZ(), hit1));
                GlobalPoint middle(hits.getValue(GlobalX(), hit2), hits.getValue(GlobalY(), hit2), hits.getValue(GlobalZ(), hit2));
                GlobalPoint outer (hits.getValue(GlobalX(), hit3), hits.getValue(GlobalY(), hit3), hits.getValue(GlobalZ(), hit3));

                //vPT[i] = length(FastHelix(outer, middle, vertex, 3.8112).getPt());
                vPT[i] = FastHelix(outer, middle, vertex, 3.8112).getPtMagnitude();
            }

            // copy vPT to tripletPT
            clever::vector<float, 1> * const tripletPt  = new clever::vector<float, 1>(vPT.size(), *contx);
            contx->transfer_to_buffer(tripletPt->get_mem(), vPT.data(), sizeof(cl_float) * tripletPt->get_count());
#else
            TrackletCircleFitter trackletCircleFitter(*contx);
            auto tripletPt = trackletCircleFitter.run(hits, *tracklets, *m_conectableTripletIndexes, exec.threads, false);
#endif
            runtime.trackletCircleFitter.stopWalltime();

            runtime.cellularAutomaton.startWalltime();
            CellularAutomaton cellularAutomaton(*contx);

            const clever::vector<uint, 1> *reconstructedTracks;
            const clever::vector<uint, 1> *reconstructedTracksOffsets;

            std::tie(reconstructedTracks, reconstructedTracksOffsets) = cellularAutomaton.run(*m_connectableTripletBasis,
                                  *m_connectableTripletFollowers,
                                  *tripletPt,
                                  *m_conectableTripletIndexes,
                                  exec.threads, false);
            runtime.cellularAutomaton.stopWalltime();

            delete m_connectableTripletBasis;
            delete m_connectableTripletFollowers;
            delete m_conectableTripletIndexes;
            delete m_nonConectableTripletIndexes;

            std::vector<uint> vTrackCollection(reconstructedTracks->get_count());
            transfer::download(*reconstructedTracks, vTrackCollection, *contx);

            std::vector<uint> vPSum(reconstructedTracksOffsets->get_count());
            transfer::download(*reconstructedTracksOffsets, vPSum, *contx);

            std::vector<float> vPt(tripletPt->get_count());
            transfer::download(*tripletPt, vPt, *contx);

            delete reconstructedTracks;
            delete reconstructedTracksOffsets;
            delete tripletPt;
#endif
            //evaluate it
            //printTracks(vTrackCollection, vPSum, tracklets, hits, vPt, runtime.tracks, dEtaCut);

            //runtime
            std::vector<tKernelEvent> runEvents;
            for (cl_event e : TripletConnectivityTight::events) {
                tKernelEvent t = (*contx).getKernelPerf(e);
                runEvents.push_back(t);
            }

            //Circle fitter
            for (cl_event e : TrackletCircleFitter::events) {
                tKernelEvent t = (*contx).getKernelPerf(e);
                runEvents.push_back(t);
            }

            //Cellular automaton
            for (cl_event e : CellularAutomaton::events) {
                tKernelEvent t = (*contx).getKernelPerf(e);
                runEvents.push_back(t);
            }
            runs.push_back(runEvents);

            runtime.fillRuntimes(*contx);

            runtime.logPrint();
            runtimeRecords.addRecord(runtime);
            //std::map<uint, std::vector<uint>> data;
            //physics
            /*
            for (uint e = 0; e < grid.config.nEvents; ++e) {
                for (uint p = 0; p < layerConfig.size(); ++p) {
                    PhysicsRecord physics(e, grid.config.nEvents, p, layerConfig.size());
                    physics.fillData(*tracklets, validTracks[e], hits, layerConfig.size(), data);
                    physicsRecords.addRecord(physics);
                }
            }
            */


/*
            for (auto it=data.begin(); it!=data.end(); ++it){
                std::cout << "TRACK " << it->first << " => ";
                auto track = it->second;
                for(auto it2 = track.begin(); it2 != track.end(); it2++){
                    std::cout << *it2 << '-';
                }
                std::cout << std::endl;
            }
*/
            //delete variables

            delete tracklets;


            //reset kernels event counts
            GridBuilder::clearEvents();
            PairGeneratorBeamspot::clearEvents();
            TripletThetaPhiPredictor::clearEvents();
            TripletThetaPhiFilter::clearEvents();
            PrefixSum::clearEvents();

#ifdef MIO
            TripletConnectivityTight::clearEvents();
            TrackletCircleFitter::clearEvents();
            CellularAutomaton::clearEvents();
#endif
            //break;
        }

        //delete edLoader;
    } //destruction order block
    return std::make_pair(runtimeRecords, physicsRecords);
}

std::string getFilename(const std::string& str)
{
    //std::cout << "Splitting: " << str << '\n';
    unsigned found = str.find_last_of("/\\");
    //std::cout << " path: " << str.substr(0,found) << '\n';
    //std::cout << " file: " << str.substr(found+1) << '\n';

    return str.substr(found + 1);
}

int main(int argc, char *argv[])
{

    namespace po = boost::program_options;

    char * chTraxDir = getenv("TRAX_DIR");
    if (chTraxDir == nullptr) {
        std::cout << "TRAX_DIR not set" << std::endl;
        exit(1);
    }
    g_traxDir = chTraxDir;

    ExecutionParameters exec;
    EventDataLoadingParameters loader;
    GridConfig grid;

    std::string testSuiteFile;

    po::options_description cLoader("Config File: Event Data Loading Options");
    cLoader.add_options()
    ("data.edSrc", po::value<std::string>(&loader.eventDataFile), "event database")
    ("data.skipEvents", po::value<uint>(&loader.skipEvents)->default_value(0),
     "events to skip, default: 0")
    ("data.maxEvents", po::value<int>(&loader.maxEvents)->default_value(1),
     "number of events to process, all events: -1")
    ("data.minPt", po::value<float>(&loader.minPt)->default_value(1.0),
     "MC data only: minimum track Pt")
    ("data.tracks", po::value<int>(&loader.maxTracks)->default_value(-1),
     "MC data only: number of valid tracks to load, all tracks: -1")
    ("data.onlyTracks", po::value<bool>(&loader.onlyTracks)->default_value(true),
     "MC data only: load only tracks with hits in each layer")
    ("data.eventLoader", po::value<std::string>(&loader.eventLoader)->default_value("standard"),
     "specify event loader: \"standard\" for PEventContainer (default); \"store\" for EventStore; \"repeated\" for performance measurements")
    ("data.singleEvent", po::value<uint>(&loader.singleEvent)->default_value(0),
     "Event to use with the SingleEventLoader")
    ;

    po::options_description cExec("Config File: Execution Options");
    cExec.add_options()
    ("exec.threads", po::value<uint>(&exec.threads)->default_value(256),
     "number of work-items in one work-group")
    ("exec.eventGrouping", po::value<uint>(&exec.eventGrouping)->default_value(1),
     "number of concurrently processed events")
    ("exec.useCPU", po::value<bool>(&exec.useCPU)->default_value(false)->zero_tokens(),
     "force using CPU instead of GPGPU")

    ("exec.computingPlatform", po::value<string>(&exec.computingPlatform)->default_value("amd"),
        "force use of particular platform [amd, nvidia, intel]")

    ("exec.iterations", po::value<uint>(&exec.iterations)->default_value(1),
     "number of iterations for performance evaluation")
    ("exec.layerTripletConfig", po::value<std::string>(&exec.layerTripletConfigFile),
     "configuration file for layer triplets")
    ("exec.layerTriplets", po::value<int>(&exec.layerTriplets)->default_value(-1),
     "number of layer triplets to load, default all: -1")
    ;

    po::options_description cGrid("Config File: Grid Options");
    cGrid.add_options()
    ("grid.nSectorsZ", po::value<uint>(&grid.nSectorsZ)->default_value(50),
     "number of grid cells in z dimension")
    ("grid.nSectorsPhi", po::value<uint>(&grid.nSectorsPhi)->default_value(8),
     "number of grid cells in phi dimension")
    ;

    po::options_description cCommandLine("Command Line Options");
    cCommandLine.add_options()
    ("config", po::value<std::string>(&exec.configFile), "configuration file")
    ("help", "produce help message")
    ("silent", "supress all messages from TripletFinder")
    ("verbose", "elaborate information")
    ("(PROLIX)", "(PROLIX) output -- degrades performance as data is transferred from device")
    ("live", "live output -- degrades performance")
    ("cpu", "force running on cpu")
    ("testSuite", po::value<std::string>(&testSuiteFile),
     "specify a file defining test cases to be run")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cCommandLine), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << cCommandLine << cExec << cGrid << cLoader << std::endl;
        return 1;
    }

    if (vm.count("config")) {
        ifstream ifs(g_traxDir + "/configs/" + getFilename(exec.configFile));
        if (!ifs) {
            cerr << "can not open config file: " << exec.configFile << "\n";
            return 0;
        } else {
            po::options_description cConfigFile;
            cConfigFile.add(cLoader);
            cConfigFile.add(cExec);
            cConfigFile.add(cGrid);
            store(parse_config_file(ifs, cConfigFile), vm);
            notify(vm);
            ifs.close();
        }
    } else {
        cerr << "No config file specified!" << std::endl;
        return 1;
    }

    //define verbosity level
    exec.verbosity = Logger::cNORMAL;
    if (vm.count("silent")) {
        exec.verbosity = Logger::cSILENT;
    }
    if (vm.count("verbose")) {
        exec.verbosity = Logger::cVERBOSE;
    }
    if (vm.count("(PROLIX)")) {
        exec.verbosity = Logger::cPROLIX;
    }
    if (vm.count("live")) {
        exec.verbosity = Logger::cLIVE;
    }
    if (vm.count("live") && vm.count("(PROLIX)")) {
        exec.verbosity = Logger::cLIVEPROLIX;
    }
    exec.verbosity = Logger::cLIVEPROLIX;
    exec.verbosity = Logger::cPROLIX;
    //check for cpu
    if (vm.count("cpu")) {
        exec.useCPU = true;
    }

    typedef std::pair<ExecutionParameters, EventDataLoadingParameters> tExecution;
    std::vector<tExecution> executions;

    //****************************************
    if (vm.count("testSuite")) {

        ifstream ifs(g_traxDir + "/configs/" + getFilename(testSuiteFile));
        if (!ifs) {
            cerr << "can not open testSuite file: " << testSuiteFile << "\n";
            return 0;
        } else {
            std::vector<uint> threads;
            std::vector<uint> events;
            std::vector<uint> tracks;

            po::options_description cTestSuite;
            cTestSuite.add_options()
            ("threads", po::value<std::vector<uint> >(&threads)->multitoken(), "work-group size")
            ("tracks", po::value<std::vector<uint> >(&tracks)->multitoken(), "tracks to load")
            ("eventGrouping", po::value<std::vector<uint> >(&events)->multitoken(),
             "events to process concurrently")
            ;

            po::variables_map tests;
            po::store(parse_config_file(ifs, cTestSuite), tests);
            po::notify(tests);
            ifs.close();

            //add default values if necessary
            if (events.size() == 0) {
                events.push_back(exec.eventGrouping);
            }
            if (threads.size() == 0) {
                threads.push_back(exec.threads);
            }
            if (tracks.size() == 0) {
                tracks.push_back(loader.maxTracks);
            }

            //add test cases
            for (uint e : events) {
                for (uint t : threads) {
                    for (uint n : tracks) {
                        ExecutionParameters ep(exec); //clone default
                        EventDataLoadingParameters lp(loader);

                        ep.threads = t;
                        ep.eventGrouping = e;
                        lp.maxEvents = e;
                        lp.maxTracks = n;

                        executions.push_back(std::make_pair(ep, lp));
                    }
                }
            }
        }
    } else {
        executions.push_back(std::make_pair(exec, loader));
    }

    //set up logger verbosity
    Logger::getInstance().setLogLevel(exec.verbosity);
    clever::context * contx = createContext(exec);

    RuntimeRecords runtimeRecords;
    PhysicsRecords physicsRecords;

    std::vector<float> dEtaCuts2;
    float begin = 0.0240;
    float end = 0.0270;
    int steps = 30;
    float step_size = (end - begin) / steps;
    for (int i = 0; i < steps; i++){
        dEtaCuts2.push_back(begin + i * step_size);
        std::cout << dEtaCuts2.back() << ' ';
    }

    for (int i = 0; i < steps; i++){
        std::cout << dEtaCuts2[i] << ' ';
    }
    //dEtaCuts2.clear();
    std::cout << std::endl;
    std::cerr << "ETACUT" << SEP << "TOTAL" << SEP << "FILTERED" << SEP << "VALIDS"  << SEP << "VALIDUNIQUE" << SEP << "FAKES"  << SEP <<  "FAKERATE"  << SEP << "CLONERATE"  << SEP <<  "EFFICIENCE"  << SEP <<  "SIMULATED" << std::endl;
    for (uint e = 0; e < executions.size(); ++e) { //standard case: only 1
        LLOG << "Experiment " << e << "/" << executions.size() << ": " << std::endl;
        LLOG << executions[e].first << " " << executions[e].second << std::endl;
        for (uint i = 0; i < exec.iterations; ++i) {
            LLOG << "Experiment iteration: " << i + 1 << "  " << std:: endl << std::flush ;

            for (float eta : dEtaCuts2){
                //cout << "dETA:" << eta << std::endl;
                auto res = buildTriplets(executions[e].first, executions[e].second, grid, contx, 0.0256);

                contx->clearAllBuffers();
                contx->clearPerfCounters();

                //runtimeRecords.merge(res.first);
                //physicsRecords.merge(res.second);
            }
        }
        LLOG << std::endl;
    }

    delete contx;
    //**********************************

    runtimeRecords.logPrint();

    std::stringstream outputFileRuntime;
    outputFileRuntime << g_traxDir << "/runtime/" << "runtime."
                      << getFilename(exec.configFile) << (testSuiteFile != "" ? "." : "")
                      << getFilename(testSuiteFile) << (exec.useCPU ? ".cpu" : ".gpu")
                      << ".csv";

    std::stringstream outputDirRuntime;
    outputDirRuntime << g_traxDir << "/runtime/" << getFilename(exec.configFile);
    boost::filesystem::create_directories(outputDirRuntime.str());

    std::ofstream runtimeRecordsFile(outputFileRuntime.str(), std::ios::trunc);
    runtimeRecordsFile << runtimeRecords.csvDump();
    runtimeRecordsFile.close();
    /*
    std::stringstream outputFilePhysics;
    outputFilePhysics << g_traxDir << "/physics/" << "physics." << getFilename(
                          exec.configFile) << ".csv";
    std::stringstream outputDirPhysics;
    outputDirPhysics << g_traxDir << "/physics/" << getFilename(exec.configFile);

    boost::filesystem::create_directories(outputDirPhysics.str());

    std::ofstream physicsRecordsFile(outputFilePhysics.str(), std::ios::trunc);
    physicsRecordsFile << physicsRecords.csvDump(outputDirPhysics.str());
    physicsRecordsFile.close();
    */
    std::cout << Logger::getInstance();
}


typedef typename std::pair<uint, std::shared_ptr<std::vector<uint>>> Track;
typedef typename std::list<Track> TrackCollection;

typedef typename std::pair<uint, std::shared_ptr<std::vector<uint>>> TripletTrack;
typedef typename std::list<TripletTrack> TripletTrackCollection;
struct TripletTrackComparator
{

    const TripletTrack &baseTrack;

    TripletTrackComparator() = delete;
    TripletTrackComparator(const TripletTrack &track) : baseTrack(track) {}

    bool operator()(const TripletTrack &other) const {
        const TripletTrack &longer = baseTrack.second->size() < other.second->size() ? other : baseTrack;
        const TripletTrack &shorter = baseTrack.second->size() >= other.second->size() ? other : baseTrack;
        uint sharedHits = 0;
        for (auto hit : *(longer.second)){
            auto found = std::find(shorter.second->begin(), shorter.second->end(), hit);
            if (found != shorter.second->end()){
                sharedHits++;
            }
        }

        const float fshared = sharedHits/float(shorter.second->size());
        return fshared >= 0.19 ? true : false;
    }
};

bool trackIsValid(const Track &track, const HitCollection &hits)
{
    const uint  hitId = hits.getValue(HitId(), track.second->front());
    for (auto hit : *(track.second)){
        if (hitId != hits.getValue(HitId(), hit)){
            return false;
        }
    }
    return true;
}

TrackCollection cloneRemoval(const TrackCollection &trackCollection)
{
    TrackCollection trackCollectionFiltered;
    for(auto &track : trackCollection){

        auto found = find_if(trackCollectionFiltered.begin(), trackCollectionFiltered.end(), TripletTrackComparator(track));

        if (found == trackCollectionFiltered.end()){
            trackCollectionFiltered.push_back(track);
        }else if (found->second->size() == track.second->size()){
            trackCollectionFiltered.push_back(track);
        }else if (found->second->size() < track.second->size()){
            /*
                std::cout << "REMOVING: ";
                for(auto hit : *found->second){
                    std::cout << hit << "-";
                }

                std::cout << std::endl;
                std::cout << "ADDING:   ";

                for(auto hit : *(i.second)){
                    std::cout << hit << "-";
                }

                std::cout << std::endl;
            */
            trackCollectionFiltered.erase(found);
            trackCollectionFiltered.push_back(track);
        }
    }
    return trackCollectionFiltered;
}

std::pair<TrackCollection, TrackCollection> splitRealFakeTracks(const TrackCollection &trackCollection, const HitCollection &hits)
{
    TrackCollection realTracks;
    TrackCollection fakeTracks;

    for(auto &track : trackCollection){
        if (trackIsValid(track, hits)){
            realTracks.push_back(track);
        }else{
            fakeTracks.push_back(track);
        }
    }
    return std::make_pair(realTracks, fakeTracks);
}

void printTrack(const Track &track, const HitCollection &hits)
{
    std::cout << "TRACK #" << track.first << " " << (trackIsValid(track, hits) ? "REAL TRACK" : "FAKE TRACK") << std::endl;
    std::cout << "    LENGTH: " << track.second->size() << std::endl;
    std::cout << "    SIM TRACK IDs: ";
    for (auto hitId : *(track.second)){
        std::cout << hits.getValue(HitId(), hitId) << " ";
    }
    std::cout << std::endl;

    std::cout << "    HIT IDS: ";
    for (auto hitId : *(track.second)){
        std::cout << hitId  << "-";
    }
    std::cout << std::endl;

    std::cout << "    HIT COORDS: " << std::endl;
    for (auto &hitId : *(track.second)){
        const auto hitX = hits.getValue(GlobalX(), hitId);
        const auto hitY = hits.getValue(GlobalY(), hitId);
        const auto hitZ = hits.getValue(GlobalZ(), hitId);
        std::cout << "        " << "[" << hitX << ", " << hitY << ", " << hitZ << "]" << std::endl;
    }
    std::cout << std::endl;
}

TrackCollection extractTracks(const std::vector<uint> &vTrackCollection,
                              const std::vector<uint> &vPSum,
                              const TrackletCollection * const tracklets,
                              const HitCollection &hits,
                              const std::vector<float> &vPt)
{
    TrackCollection trackCollectionList;
    uint trackID = 0;
    for (uint i = 0; i < vPSum.size() - 1; i++) {

        uint trackOffset = vPSum[i];
        uint trackLength = vPSum[i + 1] - trackOffset;

        if (trackLength < 3) {
            continue;
        }

        shared_ptr<std::vector<uint>> track(new std::vector<uint>());
        for (uint j = 0; j < trackLength; j++) {
            uint triplet = vTrackCollection[trackOffset + (trackLength - 1 - j)];
            uint hit1 = tracklets->getValue(TrackletHit1(), triplet);
            uint hit2 = tracklets->getValue(TrackletHit2(), triplet);
            uint hit3 = tracklets->getValue(TrackletHit3(), triplet);
            if(track->size() == 0){
                track->push_back(hit1);
                track->push_back(hit2);
            }
            track->push_back(hit3);
        }

        trackCollectionList.push_back(std::make_pair(trackID++, track));
        /*
        std::cout << "RAW TRACK: ";
        for (auto it : *track){
            PLOG << it << "-";
        }
        std::cout << std::endl;
        */
    }
    return trackCollectionList;
}

std::map<uint, Track> findUniqueRealTracks(const TrackCollection &reconstructedTracks, const HitCollection &hits)
{
    std::map<uint, Track> foundTracks;
    for (auto &track : reconstructedTracks){
        if (trackIsValid(track, hits)){
            foundTracks[hits.getValue(HitId(), track.second->front())] = track;
        }
    }
    return foundTracks;
}

void printTracks(const std::vector<uint> &vTrackCollection,
                              const std::vector<uint> &vPSum,
                              const TrackletCollection * const tracklets,
                              const HitCollection &hits,
                              const std::vector<float> &vPt,
                              uint totalTracks,
                              float dEtaCut)
{
    TrackCollection reconstructedTracks = extractTracks(vTrackCollection, vPSum, tracklets, hits, vPt);
    TrackCollection filteredTracks = cloneRemoval(reconstructedTracks);
    TrackCollection realTracks;
    TrackCollection fakeTracks;
    std::tie(realTracks, fakeTracks) = splitRealFakeTracks(filteredTracks, hits);
    std::map<uint, Track> foundUniqueRealTracks = findUniqueRealTracks(realTracks, hits);

    uint nTracks = 1;
/*
    std::cout << "RECONSTRUCTED TRACKS: " << std::endl;
    nTracks = 1;
    for(auto track : reconstructedTracks){
        std::cout <<  "    #" << nTracks++ << ": ";
        printTrack(track, hits);
    }

    std::cout << "FILTERED TRACKS: " << std::endl;
    nTracks = 1;
    for(auto &track : filteredTracks){
        std::cout <<  "    #" << nTracks++ << ": ";
        printTrack(track, hits);
    }


    std::cout << "UNIQUE TRACKS: " << std::endl;
    nTracks = 1;
    for(auto &track : filteredTracks){
        std::cout <<  "    #" << nTracks++ << ": ";
        printTrack(track, hits);
    }
*/
    std::cout << "       TOTAL: " << reconstructedTracks.size() << std::endl;
    std::cout << "    FILTERED: " << filteredTracks.size() << std::endl;
    std::cout << "      VALIDS: " << realTracks.size() << std::endl;
    std::cout << "VALID UNIQUE: " << foundUniqueRealTracks.size() << std::endl;
    std::cout << "       FAKES: " << fakeTracks.size() << std::endl;
    std::cout << "   FAKE RATE: " << fakeTracks.size() / float(filteredTracks.size()) << std::endl;
    std::cout << "  CLONE RATE: " << (realTracks.size() - foundUniqueRealTracks.size())/ float(filteredTracks.size()) << std::endl;
    std::cout << "  EFFICIENCE: " << foundUniqueRealTracks.size() / float(totalTracks) << std::endl;
    std::cout << "   SIMULATED: " << totalTracks << std::endl;

    std::cerr << dEtaCut << SEP
              << reconstructedTracks.size() << SEP
              << filteredTracks.size() << SEP
              << realTracks.size() << SEP
              << foundUniqueRealTracks.size() << SEP
              << fakeTracks.size() << SEP
              << fakeTracks.size() / float(filteredTracks.size()) << SEP
              << (realTracks.size() - foundUniqueRealTracks.size())/ float(filteredTracks.size()) << SEP
              << foundUniqueRealTracks.size() / float(totalTracks) << SEP
              <<  totalTracks << std::endl;
/*
    for(auto tripletTrack : trackCollectionListFiltered){
        if (trackIsFake(track))
    }
*/
    /*
    uint monotonicSuccess = 0;
    uint nonMonotonicSuccess = 0;
    uint monotonicFakes = 0;
    uint nonMonotonicFakes = 0;

    for (auto track : trackCollectionListFiltered) {
        iTrack++;
        //the triplet is a handler.
        PLOG << "Track #" << track.first
             << " with triplet length " << track.second->size()
             << std::endl;
        PLOG << "\tTriplets: ";


        int eventNumber = 0;
        bool monotonicDecreasingPt = true;
        float pts[20];
        float avgPt;

        for (auto hit : *(track.second)) {
            uint hit1 = tracklets->getValue(TrackletHit1(), hit);
            uint hit2 = tracklets->getValue(TrackletHit2(), hit);
            uint hit3 = tracklets->getValue(TrackletHit3(), hit);

            eventNumber = hits.getValue(EventNumber(), hit1);

            PLOG << "[" << hit1 << "-"<< hit2 << "-"<< hit3 << "]";
            PLOG << " Pt = " << vPt[triplet] << " ";
            //monotonicDecreasingPt &= vPt[triplet] <=  previousPt ||
            //    (std::abs(vPt[triplet] - previousPt)/(std::max(std::abs(vPt[triplet]), std::abs(previousPt)))) < 0.1;
            pts[j] = vPt[triplet];
        }

        avgPt = 0;
        for (uint j = 0; j < trackLength; j++) {
            PLOG << pts[j] << " ";
            avgPt += pts[j];
        }
        avgPt /= trackLength;

        float sum_suared_diffs = 0;
        for (uint j = 0; j < trackLength; j++) {
            sum_suared_diffs += std::pow(pts[j] -  avgPt, 2);
        }

        PLOG << std::endl << "\tEvent Number #: " << eventNumber << std::endl;
        PLOG << "\tTrack IDs #: ";
        bool same = true;
        uint trackId;

        for (auto it = trackIDs.begin(); it != trackIDs.end(); it++){
            if (*it != *trackIDs.begin()){
                same = false;
            }
            trackId = *trackIDs.begin();
        }

        for (auto it = trackIDs.begin(); it != trackIDs.end(); it++){
            PLOG << *it << " ";
        }

        if (same ){ //&& ((sum_suared_diffs/ trackLength) < 1.E4)){
            foundTracks[trackId] = true;
            if(monotonicDecreasingPt){
                monotonicSuccess++;
                PLOG << " =====================> RECONSTRUCTED-SUCCESS-MONOTONIC ";

            }else{
                nonMonotonicSuccess++;
                PLOG << " =====================> RECONSTRUCTED-SUCCESS-NON-MONOTONIC ";
            }
        }else{
            if(monotonicDecreasingPt) {
                PLOG << " =====================> RECONSTRUCTED-FAILURE-MONOTONIC ";
                monotonicFakes++;
            }else{
                PLOG << " =====================> RECONSTRUCTED-FAILURE-NON-MONOTONIC ";
                nonMonotonicFakes++;
            }
        }

        PLOG << "STDDEV(p_t) = " << sum_suared_diffs/ trackLength;
        if ((sum_suared_diffs/ trackLength) > 1.E4){
            PLOG << "WOULD DELETE";
        }

        trackIDs.clear();
        PLOG << std::endl;

        PLOG << "\tHits X: ";
        for (uint j = 0; j < trackLength; j++) {
            uint hit1 = tracklets->getValue(TrackletHit1(), vTrackCollection[trackOffset + (trackLength - 1 - j)]);
            uint hit2 = tracklets->getValue(TrackletHit2(), vTrackCollection[trackOffset + (trackLength - 1 - j)]);
            uint hit3 = tracklets->getValue(TrackletHit3(), vTrackCollection[trackOffset + (trackLength - 1 - j)]);

            PLOG << "["  << hits.getValue(GlobalX(), hit1)
                 << ", " << hits.getValue(GlobalX(), hit2)
                 << ", " << hits.getValue(GlobalX(), hit3)
                 << "]";
        }
        PLOG << std::endl << std::endl;
    }

    int trackNumber = 1;
    for (auto it = foundTracks.begin(); it != foundTracks.end(); it++){
        PLOG << "#" << trackNumber <<" UNIQUE " << it->first << std::endl;
        trackNumber++;
    }

    PLOG << "Sucessful reconstructions: " << monotonicSuccess + nonMonotonicSuccess
        << "(" << monotonicSuccess << ", " << nonMonotonicSuccess <<")" << std::endl;
    PLOG << "Fake reconstructions: " << monotonicFakes + nonMonotonicFakes
        << "(" << monotonicFakes << ", " << nonMonotonicFakes <<")" << std::endl;
    */
}
