#pragma once

//#include <tuple>
#include <utility>
#include <vector>
#include <boost/type_traits.hpp>

#include <clever/clever.hpp>

#include "CommonTypes.h"

struct TrackletHit1: public clever::UIntItem
{
};
struct TrackletHit2: public clever::UIntItem
{
};
struct TrackletHit3: public clever::UIntItem
{
};

struct TrackletId: public clever::UIntItem
{
};

#define TRACKLET_COLLECTION_ITEMS TrackletId, TrackletHit1, TrackletHit2,TrackletHit3

typedef clever::Collection<TRACKLET_COLLECTION_ITEMS> TrackletCollectiontems;

class TrackletCollection: public TrackletCollectiontems
{
public:
	typedef TrackletCollectiontems dataitems_type;

	TrackletCollection()
	{

	}

	TrackletCollection(int items) :
			clever::Collection<TRACKLET_COLLECTION_ITEMS>(items)
	{

	}
};

typedef clever::OpenCLTransfer<TRACKLET_COLLECTION_ITEMS> TrackletCollectionTransfer;

class Tracklet: private clever::CollectionView<TrackletCollection>
{
public:
// get a pointer to one hit in the collection
	Tracklet(TrackletCollection & collection, index_type i) :
			clever::CollectionView<TrackletCollection>(collection, i)
	{

	}

// create a new hit in the collection
	Tracklet(TrackletCollection & collection) :
			clever::CollectionView<TrackletCollection>(collection)
	{
	}

	float hit1() const
	{
		return getValue<TrackletHit1>();
	}

	float hit2() const
	{
		return getValue<TrackletHit2>();
	}

	float hit3() const
	{
		return getValue<TrackletHit3>();
	}

	float id() const {
		return getValue<TrackletId>();
	}

};

