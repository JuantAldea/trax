export TRAX_DIR=/afs/cern.ch/user/j/jaldeaar/WORKSPACE/GPU/trax
export LD_LIBRARY_PATH=/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/lib64/:/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/lib/
export CXX=/afs/cern.ch/cms/slc6_amd64_gcc472/external/gcc/4.7.2/bin/g++
mkdir build
cd build
cmake ../src/
make -j8
