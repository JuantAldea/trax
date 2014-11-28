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
void printTracks(const std::vector<uint> &vTrackCollection, const std::vector<uint> &vPSum,
        const TrackletCollection * const tracklets, const HitCollection &hits, const std::vector<float> &vPt);
clever::context * createContext(ExecutionParameters exec)
{
    LLOG << "Creating context for " << (exec.useCPU ? "CPU" : "GPGPU") << "...";

//#define DEBUG
#define CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL  (1<<31)
#if defined(DEBUG) && defined(CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL)
#define DEBUG_OCL CL_QUEUE_THREAD_LOCAL_EXEC_ENABLE_INTEL
#else
#define DEBUG_OCL 0
#endif

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
        EventDataLoadingParameters loader, GridConfig gridConfig, clever::context * contx)
{
    RuntimeRecords runtimeRecords;
    PhysicsRecords physicsRecords;

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
        EventLoader * edLoader = EventLoaderFactory::create(loader.eventLoader, loader);

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

            LOG << "Loaded " << hits.size() << " hits in " << evtGroupSize << " events" << std::endl;

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

            runtime.buildGrid.startWalltime();
            gridBuilder.run(hits, exec.threads, *eventSupplement, *layerSupplement, grid);
            runtime.buildGrid.stopWalltime();

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

            runtime.pairGen.startWalltime();
            Pairing * pairs = pairGen.run(hits, geomSupplement, exec.threads, layerConfig, grid);
            runtime.pairGen.stopWalltime();

            TripletThetaPhiPredictor predictor(*contx);

            runtime.tripletPredict.startWalltime();
            Pairing * tripletCandidates = predictor.run(hits, geom, geomSupplement, dict, exec.threads,
                                          layerConfig, grid, *pairs, false);
            runtime.tripletPredict.stopWalltime();

            TripletThetaPhiFilter tripletThetaPhi(*contx);

            runtime.tripletFilter.startWalltime();
            TrackletCollection * tracklets = tripletThetaPhi.run(hits, grid, *pairs, *tripletCandidates,
                                             exec.threads, layerConfig, false);
            runtime.tripletFilter.stopWalltime();

            /*******************************************/
#define MIO
#ifdef MIO
            std::map<uint, std::vector<uint>> tracks;
            for (uint i = 0; i < hits.size(); i++){
                tracks[hits.getValue(HitId(), i)].push_back(i);
                std::sort(tracks[hits.getValue(HitId(), i)].begin(), tracks[hits.getValue(HitId(), i)].end());
            }

            int trackNumber = 1;
            for (auto it=tracks.begin(); it!=tracks.end(); ++it){
                PLOG << "#" << trackNumber << " SIMULATED " << it->first << " => ";
                trackNumber++;
                auto track = it->second;
                for(auto it2 = track.begin(); it2 != track.end(); it2++){
                    PLOG << *it2 << '-';
                }
                PLOG << std::endl;
            }

            // On average a dEta = 0.0256 cut reduces the amount of triplets pairs by a 33.7% (stdev 19.6)
            TripletConnectivityTight tripletConnectivityTight(*contx);
            //float dEtaCut  = 0.0256;
            float dEtaCut  = 0.0256;
            runtime.tripletConnectivity.startWalltime();
            auto connectableTrackletsPairIndices = tripletConnectivityTight.run(hits, *tracklets, layerConfig, dEtaCut,
                                                   exec.threads, false);
            runtime.tripletConnectivity.stopWalltime();

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
            auto tripletPt = trackletCircleFitter.run(hits, *tracklets,
                             *std::get<2>(connectableTrackletsPairIndices), exec.threads, false);
            #endif

            runtime.trackletCircleFitter.stopWalltime();

            runtime.cellularAutomaton.startWalltime();
            CellularAutomaton cellularAutomaton(*contx);
            auto caOutput = cellularAutomaton.run(*std::get<0>(connectableTrackletsPairIndices),
                                  *std::get<1>(connectableTrackletsPairIndices),
                                  *tripletPt,
                                  *std::get<2>(connectableTrackletsPairIndices),
                                  exec.threads, true);
            runtime.cellularAutomaton.stopWalltime();


            std::vector<uint> vTrackCollection(std::get<0>(caOutput)->get_count());
            transfer::download(*std::get<0>(caOutput), vTrackCollection, *contx);

            std::vector<uint> vPSum(std::get<1>(caOutput)->get_count());
            transfer::download(*std::get<1>(caOutput), vPSum, *contx);

            std::vector<float> vPt(tripletPt->get_count());
            transfer::download(*tripletPt, vPt, *contx);

            //printTracks(vTrackCollection, vPSum, tracklets, hits, vPt);


            delete std::get<0>(connectableTrackletsPairIndices);
            delete std::get<1>(connectableTrackletsPairIndices);
            delete std::get<2>(connectableTrackletsPairIndices);
            delete std::get<3>(connectableTrackletsPairIndices);
            delete std::get<0>(caOutput);
            delete std::get<1>(caOutput);

            LOG << std::endl;
#endif
            //evaluate it

            //runtime
            runtime.fillRuntimes(*contx);
            runtime.logPrint();
            runtimeRecords.addRecord(runtime);
            std::map<uint, std::vector<uint>> data;
            //physics
            for (uint e = 0; e < grid.config.nEvents; ++e) {
                for (uint p = 0; p < layerConfig.size(); ++p) {
                    PhysicsRecord physics(e, grid.config.nEvents, p, layerConfig.size());
                    physics.fillData(*tracklets, validTracks[e], hits, layerConfig.size(), data);
                    physicsRecords.addRecord(physics);
                }
            }
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
            delete pairs;
            delete tripletCandidates;
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

        delete edLoader;
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

    for (uint e = 0; e < executions.size(); ++e) { //standard case: only 1
        LLOG << "Experiment " << e << "/" << executions.size() << ": " << std::endl;
        LLOG << executions[e].first << " " << executions[e].second << std::endl;
        for (uint i = 0; i < exec.iterations; ++i) {
            LLOG << "Experiment iteration: " << i + 1 << "  " << std::flush;
            auto res = buildTriplets(executions[e].first, executions[e].second, grid, contx);

            contx->clearAllBuffers();
            contx->clearPerfCounters();

            runtimeRecords.merge(res.first);
            physicsRecords.merge(res.second);
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

    std::stringstream outputFilePhysics;
    outputFilePhysics << g_traxDir << "/physics/" << "physics." << getFilename(
                          exec.configFile) << ".csv";
    std::stringstream outputDirPhysics;
    outputDirPhysics << g_traxDir << "/physics/" << getFilename(exec.configFile);

    boost::filesystem::create_directories(outputDirPhysics.str());

    std::ofstream physicsRecordsFile(outputFilePhysics.str(), std::ios::trunc);
    physicsRecordsFile << physicsRecords.csvDump(outputDirPhysics.str());
    physicsRecordsFile.close();

    std::cout << Logger::getInstance();
}

void printTracks(const std::vector<uint> &vTrackCollection,
    const std::vector<uint> &vPSum,
    const TrackletCollection * const tracklets,
    const HitCollection &hits,
    const std::vector<float> &vPt)
{
    int iTrack = 0;

    std::map<uint, bool> foundTracks;

    std::deque<std::vector<uint>*> trackCollection;
    for (uint i = 0; i < vPSum.size() - 1; i++) {

        uint trackOffset = vPSum[i];
        uint trackLength = vPSum[i + 1] - trackOffset;

        if (trackLength < 3) {
            continue;
        }

        std::vector<uint>* track = new std::vector<uint>();
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
        trackCollection.push_back(track);
    }

    std::set<uint> tracksToRemove;

    for (uint i = 0; i < trackCollection.size(); i++){
        std::vector<uint> *track = trackCollection[i];
        if(tracksToRemove.find(i) != tracksToRemove.end()){
            continue;
        }


        for(uint j = i + 1; j < trackCollection.size(); j++){
            std::vector<uint> *track2 = trackCollection[j];
            if(tracksToRemove.find(j) != tracksToRemove.end()){
                continue;
            }

            if(tracksToRemove.find(i) != tracksToRemove.end()){
                break;
            }

            if (track2->size() == track->size()){
                continue;
            }
            std::vector<uint> *longerTrack;
            std::vector<uint> *shorterTrack;
            uint removed;
            uint remover;
            if(track2->size() > track->size()){
                longerTrack = track2;
                shorterTrack = track;
                removed = i;
                remover = j;
            }else {
                longerTrack = track;
                shorterTrack = track2;
                removed = j;
                remover = i;
            }
            uint sharedHits = 0;
            for (auto hit : *shorterTrack){
                if (std::find(longerTrack->begin(), longerTrack->end(), hit) != longerTrack->end()){
                    sharedHits++;
                }
            }
            float fshared = sharedHits/float(shorterTrack->size());
            if (fshared >= 0.19){
                PLOG << "FSHARED " << fshared << std::endl;;
                tracksToRemove.insert(removed);
                PLOG << "REMOVED-SHARED " <<  remover << " removes " << removed << std::endl;
                for (auto hit : *track){
                    PLOG << hit << '-';
                }
                PLOG << std::endl;
                for (auto hit : *track2){
                    PLOG << hit << '-';
                }
                PLOG << std::endl;
            }

        }
    }


    for(auto it = tracksToRemove.begin(); it != tracksToRemove.end(); it++){
        //trackCollection.erase(std::find(trackCollection.begin(), trackCollection.end(), *it));
        PLOG << "REMOVED" << *it << std::endl;
    }

    uint monotonicSuccess = 0;
    uint nonMonotonicSuccess = 0;
    uint monotonicFakes = 0;
    uint nonMonotonicFakes = 0;

    for (uint i = 0; i < vPSum.size() - 1; i++) {

        uint trackOffset = vPSum[i];
        uint trackLength = vPSum[i + 1] - trackOffset;

        if (trackLength < 3) {
            continue;
        }
        PLOG << "buscando " << i << std::endl;
        if (std::find(tracksToRemove.begin(), tracksToRemove.end(), uint(i)) != tracksToRemove.end()){
            PLOG << "Track #" << iTrack << " was deleted" << std::endl;
            continue;
        }

        iTrack++;
        //the triplet is a handler.
        PLOG << "Track #" << iTrack
             << " with triplet length " << trackLength
             << " begins at " << trackOffset
             << std::endl;
        PLOG << "\tTriplets: ";

        std::vector<uint> trackIDs;
        bool monotonicDecreasingPt = true;
        float previousPt = std::numeric_limits<float>::max();
        float pts[20];
        float avgPt;

        for (uint j = 0; j < trackLength; j++) {
            uint triplet = vTrackCollection[trackOffset + (trackLength - 1 - j)];
            uint hit1 = tracklets->getValue(TrackletHit1(), triplet);
            uint hit2 = tracklets->getValue(TrackletHit2(), triplet);
            uint hit3 = tracklets->getValue(TrackletHit3(), triplet);

            trackIDs.push_back(hits.getValue(HitId(), hit1));
            trackIDs.push_back(hits.getValue(HitId(), hit2));
            trackIDs.push_back(hits.getValue(HitId(), hit3));

            PLOG << "[" << hit1 << "-"<< hit2 << "-"<< hit3 << "]";
            // Ideal Magnetic Field [T] (-0,-0, 3.8112)
            const float BZ = 3.8112;
            // e = 1.602177×10^-19 C (coulombs)
            const float Q = 1.602177E-19;
            // c = 2.998×10^8 m/s (meters per second)
            //const float C = 2.998E8;
            // 1 GeV/c = 5.344286×10^-19 J s/m (joule seconds per meter)
            const float GEV_C = 5.344286E-19;
            //PLOG << " Pt = " << Q * BZ * (vPt[triplet] * 1E-2) / GEV_C << " ";
            PLOG << " Pt = " << vPt[triplet] << " ";
            //monotonicDecreasingPt &= vPt[triplet] <=  previousPt ||
            //    (std::abs(vPt[triplet] - previousPt)/(std::max(std::abs(vPt[triplet]), std::abs(previousPt)))) < 0.1;
            pts[j] = vPt[triplet];
            previousPt = vPt[triplet];
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

            PLOG << "[" << hits.getValue(GlobalX(), hit1)
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
}
