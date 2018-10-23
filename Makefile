### command and flags ###
TARGET = eclairs
PY_TARGET = pyeclairs


### C++ compiler ###
#CXX = g++
CXX = icpc
#CXX = clang++

CXXFLAGS = -O2 -Wall -std=c++11
#CXXFLAGS = -O0 -g -std=c+11

SRCDIR   = ./src
GSL_DIR  = /opt/local
INCLUDES = -I$(GSL_DIR)/include
LIBS     = -L$(GSL_DIR)/lib -lgsl -lgslcblas -lm


### Python module setting ###
# detect OS
OS := $(shell uname)
$(info OS:$(OS))

# For macOS with MacPorts
ifeq ($(OS),Darwin)
	BOOST_DIR = /opt/local
	PY_VERSION = 2.7
	PY_DIR = /opt/local/Library/Frameworks/Python.framework/Versions/$(PY_VERSION)
	PY_INCLUDES = -I$(PY_DIR)/include/python$(PY_VERSION) -I$(BOOST_DIR)/include -I$(GSL_DIR)/include
	PY_LIBS = -L$(PY_DIR)/lib/python$(PY_VERSION) -L$(BOOST_DIR)/lib -L$(GSL_DIR)/lib \
	-lgsl -lgslcblas -lboost_python-mt -lboost_numpy-mt -lpython$(PY_VERSION)
	PY_FLAGS = $(CXXFLAGS) -fPIC -shared
endif

# For generic Linux system with anaconda
ifeq ($(OS),Linux)
	BOOST_DIR = /work/osatokn/usr/local
	PY_VERSION = 2.7
	PY_DIR = /work/osatokn/anaconda2/
	PY_INCLUDES = -I$(PY_DIR)/include/python$(PY_VERSION) -I$(BOOST_DIR)/include -I$(GSL_DIR)/include
	PY_LIBS = -L$(PY_DIR)/lib/python$(PY_VERSION) -L$(BOOST_DIR)/lib -L$(GSL_DIR)/lib \
	-lgsl -lgslcblas -lboost_python -lboost_numpy -lpython$(PY_VERSION)
	PY_FLAGS = $(CXXFLAGS) -fPIC -shared
endif

# If your OS is not macOS nor Linux, please provide paths and version related with python manually.


SRCS = vector.cpp kernel.cpp Gamma1.cpp Gamma2.cpp io.cpp cosmology.cpp spectra.cpp main.cpp
SRCH = vector.hpp kernel.hpp io.hpp cosmology.hpp spectra.hpp
PY_SRCS = vector.cpp io.cpp cosmology.cpp Gamma1.cpp Gamma2.cpp kernel.cpp spectra.cpp pyeclairs.cpp
OBJS := $(SRCS:.cpp=.o)
PY_OBJS := $(PY_SRCS:.cpp=.o)


$(TARGET): $(OBJS) $(SRCH)
	$(CXX) $(CXXFLAGS) $(OBJS) $(LIBS) -o $@

$(PY_TARGET): $(PY_OBJS) $(SRCH)
	$(CXX) $(PY_FLAGS) $(PY_OBJS) $(PY_INCLUDES) $(PY_LIBS) -o $(PY_TARGET).so

pyeclairs.o: pyeclairs.cpp $(SRCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PY_INCLUDES) -c $<

.cpp.o: $(SRCH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<


.PHONY: all clean

all: $(TARGET) $(PY_TARGET)

clean:
	rm -f $(TARGET) $(PY_TARGET).so *.o
