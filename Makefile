.PHONY: all clean benchmark triblock_test

OUT_DIR := ./out
CUDAFLAGS := -Wno-deprecated-gpu-targets -arch=sm_89 -lcublas -lcusolver
ifeq ($(DEBUG),1)
	CUDAFLAGS += -G -DDEBUG
else
	CUDAFLAGS += -Xptxas -O3
endif

all: benchmark triblock_test

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%: %.cu | $(OUT_DIR)
	nvcc $< -o $@ $(CUDAFLAGS)
	cuobjdump -sass $@ > $(OUT_DIR)/$*.sass

benchmark: $(OUT_DIR)/benchmark
	$(OUT_DIR)/benchmark


triblock_test: $(OUT_DIR)/triblock_test
	$(OUT_DIR)/triblock_test

clean:
	rm -rf $(OUT_DIR)
	rm -rf telerun-out
