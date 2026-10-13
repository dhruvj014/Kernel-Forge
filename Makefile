# Linux equivalent of make.ps1. Assumes CUDA toolkit on PATH and a .venv
# created with: python -m venv .venv && make setup
PY := .venv/bin/python
export HF_HOME := $(CURDIR)/.cache/hf
export TORCH_CUDA_ARCH_LIST ?= 8.9

.PHONY: setup build test bench report ncu lint clean

setup:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-build-isolation --no-deps -e .

build:
	mkdir -p build
	$(PY) setup.py build_ext --inplace 2>&1 | tee build/ext_build.log
	.venv/bin/cmake -S . -B build/cmake -G Ninja -DCMAKE_BUILD_TYPE=Release
	.venv/bin/cmake --build build/cmake 2>&1 | tee build/cmake_build.log
	$(PY) scripts/ptxas_summary.py build/ext_build.log build/cmake_build.log

test:
	$(PY) -m pytest tests

bench:
	$(PY) bench/hw_ceilings.py
	$(PY) bench/bench_attention.py

report:
	$(PY) scripts/make_report.py

ncu:
	bash scripts/run_ncu.sh

lint:
	$(PY) -m ruff check .

clean:
	rm -rf build kernelforge/_C*.so
