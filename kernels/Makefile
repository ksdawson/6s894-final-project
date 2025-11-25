.PHONY: all clean trsm

OUT_DIR := ./out
CUDAFLAGS := -Wno-deprecated-gpu-targets -arch=sm_89
ifeq ($(DEBUG),1)
	CUDAFLAGS += -G -DDEBUG
else
	CUDAFLAGS += -Xptxas -O3
endif

all: trsm

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%: %.cu | $(OUT_DIR)
	nvcc $< -o $@ $(CUDAFLAGS)
	cuobjdump -sass $@ > $(OUT_DIR)/$*.sass

trsm: $(OUT_DIR)/trsm
	$(OUT_DIR)/trsm

clean:
	rm -rf $(OUT_DIR)
	rm -rf telerun-out
