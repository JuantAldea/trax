#pragma once

#include <iostream>
#include <fcntl.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <datastructures/serialize/Event.pb.h>
#include <datastructures/serialize/PEventStore.h>
#include <datastructures/Logger.h>

#include "Parameters.h"

class EventLoader{

public:

	EventLoader(EventDataLoadingParameters config ){
		GOOGLE_PROTOBUF_VERIFY_VERSION;
		params = config;
		read = config.skipEvents;

		std::stringstream f;
		f << getenv("TRAX_DIR") << "/data/" << config.eventDataFile;

		int fd = open(f.str().c_str(), O_RDONLY);
		google::protobuf::io::FileInputStream fStream(fd);
		google::protobuf::io::CodedInputStream cStream(&fStream);

		cStream.SetTotalBytesLimit(536870912, -1);

		if(!pContainer.ParseFromCodedStream(&cStream)){
			std::cerr << "Could not read protocol buffer" << std::endl;
			return;
		} else {
			LOG << "Opened file " << config.eventDataFile << " with " << pContainer.events_size() << " events" << std::endl;
		}
		cStream.~CodedInputStream();
		fStream.Close();
		fStream.~FileInputStream();
		close(fd);
	}


	virtual ~EventLoader()
	{
		google::protobuf::ShutdownProtobufLibrary();
	}

	virtual int nEvents() const {
		return pContainer.events_size();
	}

	virtual const PB_Event::PEvent & getEvent() const {
		return pContainer.events(read++);
	}


protected:
	EventLoader() : read(0) { }

	EventDataLoadingParameters params;

	PB_Event::PEventContainer pContainer;
	mutable uint read;
};

class CyclicEventLoader : public EventLoader
{
public:
	CyclicEventLoader(EventDataLoadingParameters config):
		EventLoader(config)
	{
		;
	}

	virtual ~CyclicEventLoader()
	{
		google::protobuf::ShutdownProtobufLibrary();
	}

	virtual int nEvents() const {
		return params.maxEvents;
	}

	virtual const PB_Event::PEvent & getEvent() const {
		auto &event = pContainer.events(read);
		read = (read + 1) % EventLoader::nEvents();
		return event;
	}

};

class SingleEventLoader : public EventLoader
{
public:
	SingleEventLoader(EventDataLoadingParameters config):
		EventLoader(config),
		event(0)
	{
		event = config.singleEvent;
	}

	virtual ~SingleEventLoader()
	{
		google::protobuf::ShutdownProtobufLibrary();
	}

	virtual int nEvents() const {
		return params.maxEvents;
	}

	virtual const PB_Event::PEvent & getEvent() const {
		return pContainer.events(event);
	}
protected:
	mutable uint event;
};

class RepeatedEventLoader : public EventLoader{

public:

	RepeatedEventLoader(EventDataLoadingParameters config ){
		params = config;

		std::stringstream f;
		f << getenv("TRAX_DIR") << "/data/" << config.eventDataFile;

		int fd = open(f.str().c_str(), O_RDONLY);
		google::protobuf::io::FileInputStream fStream(fd);
		google::protobuf::io::CodedInputStream cStream(&fStream);

		cStream.SetTotalBytesLimit(536870912, -1);

		PB_Event::PEventContainer pContainer;
		if(!pContainer.ParseFromCodedStream(&cStream)){
			std::cerr << "Could not read protocol buffer" << std::endl;
			return;
		} else {
			LOG << "Opened file " << config.eventDataFile << " with " << pContainer.events_size() << " events" << std::endl;
		}
		cStream.~CodedInputStream();
		fStream.Close();
		fStream.~FileInputStream();
		close(fd);

		event = pContainer.events(config.skipEvents);

	}


	virtual ~RepeatedEventLoader()
	{
		google::protobuf::ShutdownProtobufLibrary();
	}

	virtual int nEvents() const {
		return params.maxEvents;
	}

	virtual const PB_Event::PEvent & getEvent() const {
		return event;
	}

private:
	PB_Event::PEvent event;
};

class EventStoreLoader : public EventLoader{

public:

	EventStoreLoader(EventDataLoadingParameters config ){
		params = config;

		std::stringstream f;
		f << getenv("TRAX_DIR") << "/data/" << config.eventDataFile;

		esIn = new EventStoreInput(f.str());

		LOG << "Opened file " << config.eventDataFile << " with " << nEvents() << " events" << std::endl;

		for(uint i = 0; i < config.skipEvents; ++i)
			esIn->readNextElement();

	}

	virtual ~EventStoreLoader(){
		esIn->Close();
		delete esIn;
		google::protobuf::ShutdownProtobufLibrary();
	}

	virtual int nEvents() const {
		return esIn->getEvents();
	}

	virtual const PB_Event::PEvent & getEvent() const {
		return *(esIn->readNextElement()); //caller needs to ensure to not to read over file limit
	}

private:
	EventStoreInput * esIn;
};

class EventLoaderFactory {

public:
	static EventLoader * create(std::string evtLoader, EventDataLoadingParameters config){

		EventLoader * loader = NULL;

		if(evtLoader == "standard")
			loader = new EventLoader(config);

		if(evtLoader == "repeated")
			loader =  new RepeatedEventLoader(config);

		if(evtLoader == "store")
			loader =  new EventStoreLoader(config);
		if(evtLoader == "cyclic")
			loader =  new CyclicEventLoader(config);
		if(evtLoader == "single")
			loader  = new SingleEventLoader(config);
		return loader;
	}

};
