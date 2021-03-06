#pragma once

#include "../HitCollection.h"

#include <iostream>
#include <fstream>

namespace HitCollectionData
{
void generatHitTestData(HitCollection & ht, const unsigned int hitCount = 20000)
{

	for (unsigned int i = 0; i < hitCount; i++)
	{
		ht.addWithValue(10.f * float(i), float(i), 1.0f / float(i+1), i % 10, i % 10, i + 1, 0);
	}
}

HitCollection::tTrackList loadHitDataFromPB(HitCollection &ht, std::string filename, const DetectorGeometry & geom, LayerSupplement & layerSupplement,
		float minPt = 0, int numTracks = -1, bool onlyTracks = false, uint maxLayer = 99) {

	HitCollection::tTrackList result;

	PB_Event::PEventContainer pContainer;
	std::fstream in(filename, std::ios::in | std::ios::binary);

	if(!pContainer.ParseFromIstream(&in)){
		std::cerr << "Could not read protocol buffer" << std::endl;
		return result;
	}

	for(auto event : pContainer.events()){
		HitCollection::tTrackList evt = ht.addEvent(event, geom, layerSupplement, minPt, numTracks, onlyTracks, maxLayer);

		result.insert(evt.begin(), evt.end());
	}

	return result;

}

}
