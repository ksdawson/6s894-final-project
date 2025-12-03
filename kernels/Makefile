.PHONY: all clean gtrsm

OUT_DIR := ./out
CUDAFLAGS := -Wno-deprecated-gpu-targets -arch=sm_89 -lcublas
ifeq ($(DEBUG),1)
	CUDAFLAGS += -G -DDEBUG
else
	CUDAFLAGS += -Xptxas -O3
endif

all: gtrsm

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%: %.cu | $(OUT_DIR)
	nvcc $< -o $@ $(CUDAFLAGS)
	cuobjdump -sass $@ > $(OUT_DIR)/$*.sass

gtrsm: $(OUT_DIR)/gtrsm
	$(OUT_DIR)/gtrsm

clean:
	rm -rf $(OUT_DIR)
	rm -rf telerun-out
