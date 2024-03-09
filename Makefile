### command and flags ###
TARGET = eclairs
PY_TARGET = pyeclairs

vpath %.cpp src
vpath %.hpp src

# For kernel precomputation, MPI parallelization is supported.
# But for other cases, this flag should not be set.
#MPI_PARALLEL=0

ifdef SYSTYPE
SYSTYPE := $(SYSTYPE)
else
SYSTYPE := $(shell uname)
endif

$(info OS:$(SYSTYPE))

ifeq ($(SYSTYPE),Darwin)
CXX = clang++
#CXXFLAGS = -O2 -Wall -fPIC -std=c++11
CXXFLAGS = -O0 -g -Wall -fPIC -std=c++11

GSL_DIR  = /opt/homebrew
BOOST_DIR = /opt/homebrew
endif

ifeq ($(SYSTYPE),XC50)
ifdef MPI_PARALLEL
CXX = CC
else
CXX = icpc
endif
CXXFLAGS = -fast -Wall -fPIC -std=c++11

GSL_DIR  = /work/osatokn/usr
BOOST_DIR = /work/osatokn/usr
endif

ifdef MPI_PARALLEL
CXXFLAGS += -DMPI_PARALLEL
endif

INCLUDES = -I$(GSL_DIR)/include
LIBS     = -L$(GSL_DIR)/lib -lm -lgsl -lgslcblas

ifdef MPI_PARALLEL
INCLUDES += -I$(BOOST_DIR)/include
LIBS     += -L$(BOOST_DIR)/lib -lboost_mpi -lboost_serialization
endif

### Python module setting ###

# For macOS with homebrew
ifeq ($(SYSTYPE),Darwin)
	PIP = pip
	PY_DIR = /opt/Library/Frameworks/Python.framework/Versions/$(PY_VERSION)
	PY_INCLUDES = -I$(PY_DIR)/include/python$(PY_VERSION)m -I$(BOOST_DIR)/include -I$(GSL_DIR)/include
	PY_LIBS = -L$(PY_DIR)/lib -L$(BOOST_DIR)/lib -L$(GSL_DIR)/lib \
	-lgsl -lgslcblas -lboost_python37-mt -lboost_numpy37-mt -lpython$(PY_VERSION)
	PY_FLAGS = $(CXXFLAGS) -fPIC -shared
endif

# For generic Linux system with anaconda
ifeq ($(SYSTYPE),XC50)
	PIP = pip
	PY_DIR = /work/osatokn/anaconda3
	PY_INCLUDES = -I$(PY_DIR)/include/python3.11 -I$(BOOST_DIR)/include -I$(GSL_DIR)/include
	PY_LIBS = -L$(PY_DIR)/lib -L$(BOOST_DIR)/lib -L$(GSL_DIR)/lib \
	-lgsl -lgslcblas -lboost_python311 -lboost_numpy311 -lpython3
	PY_FLAGS = $(CXXFLAGS) -fPIC -shared
endif

# If your OS is not macOS nor Linux, please provide paths and version related with python manually.



SRCS = vector.cpp kernel.cpp Gamma1.cpp Gamma2.cpp params.cpp cosmology.cpp spectra.cpp nonlinear.cpp \
       bispectra.cpp direct_red.cpp fast_kernels.cpp fast_spectra.cpp \
       fast_kernels_bispec.cpp fast_bispectra.cpp spectra_red.cpp IR_EFT.cpp main.cpp
SRCH = vector.hpp kernel.hpp params.hpp cosmology.hpp spectra.hpp nonlinear.hpp \
       bispectra.hpp direct_red.hpp fast_kernels.hpp fast_spectra.hpp \
       fast_kernels_bispec.hpp fast_bispectra.hpp spectra_red.hpp IR_EFT.hpp
PY_SRCS = vector.cpp params.cpp Gamma1.cpp Gamma2.cpp cosmology.cpp kernel.cpp spectra.cpp \
          bispectra.cpp direct_red.cpp fast_kernels.cpp fast_spectra.cpp spectra_red.cpp IR_EFT.cpp pyeclairs.cpp
OBJS := $(SRCS:.cpp=.o)
PY_OBJS := $(PY_SRCS:.cpp=.o)


$(TARGET): $(OBJS) $(SRCH)
	$(CXX) $(CXXFLAGS) $(addprefix build/,$(OBJS)) $(LIBS) -o $@

$(PY_TARGET): $(PY_OBJS) $(SRCH)
	$(CXX) $(PY_FLAGS) $(addprefix build/,$(PY_OBJS)) $(PY_INCLUDES) $(PY_LIBS) -o python/pyecl/$(PY_TARGET).so
	cd python; $(PIP) install setup.py

pyeclairs.o: pyeclairs.cpp $(SRCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PY_INCLUDES) -c $< -o build/$*.o

.cpp.o: $(SRCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o build/$*.o

.PHONY: all clean

all: $(TARGET) $(PY_TARGET)

clean:
	rm -f $(TARGET) build/*.o python/pyecl/$(PY_TARGET).so

