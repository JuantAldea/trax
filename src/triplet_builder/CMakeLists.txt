cmake_minimum_required (VERSION 2.6)
project (trax)

add_executable(tripletBuilder
	TripletBuilder.cpp
)

target_link_libraries(tripletBuilder datastructures)
target_link_libraries(tripletBuilder algorithms)

target_link_libraries(tripletBuilder "OpenCL")
target_link_libraries(tripletBuilder "boost_program_options")