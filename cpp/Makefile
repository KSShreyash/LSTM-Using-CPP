# ──────────────────────────────────────────────────
#  Makefile for the modular C++ LSTM trainer
#  Usage:  make           (build)
#          make clean     (remove build artefacts)
# ──────────────────────────────────────────────────

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Iinclude

TARGET   := lstm_train
SRCDIR   := src
OBJDIR   := build

SRCS := $(wildcard $(SRCDIR)/*.cpp)
OBJS := $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))

.PHONY: all clean

all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)