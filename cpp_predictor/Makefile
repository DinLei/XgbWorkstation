CXX      := gcc -std=c++11
#CXXFLAGS := -pedantic-errors -Wall -Wextra -Werror
CXXFLAGS := -w -O3 -W -Wall
LDFLAGS  := -L/usr/lib -lstdc++ -lm
BUILD    := ./build
OBJ_DIR  := $(BUILD)/obj
APP_DIR  := $(BUILD)/exe
TARGET   := xbg_pred
INCLUDE  := -Iinclude/
SRC      :=                        \
   $(wildcard test/*.cc)                                 \
# SRC      :=                      \
#    $(wildcard src/module1/*.cpp) \
#    $(wildcard src/module2/*.cpp) \
#    $(wildcard src/module3/*.cpp) \
#    $(wildcard src/*.cpp)         \

OBJECTS  := $(SRC:%.cc=$(OBJ_DIR)/%.o)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@ $(LDFLAGS)

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

.PHONY: all build clean debug release

build:
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)
	-@rm -rvf $(APP_DIR)

# the -o $@ says to put the output of the compilation in the file named on the left side of the :
# the $< is the first item in the dependencies list

# 1、wildcard : 扩展通配符
# 2、notdir ： 去除路径
# 3、patsubst ：替换通配符


# make all
# make clean
# make program
# make build

# make release
# make debug