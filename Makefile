SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

STEPPING_DIR := Stepping
PY := python3

YARP_PORT ?= 11000

YARP_SERVER := yarpserver --write
GAIT_CMD := cd $(STEPPING_DIR) && $(PY) gait.py
SENDER_CMD := cd $(STEPPING_DIR) && $(PY) Sender.py
VIZ_CMD := cd $(STEPPING_DIR) && $(PY) visulizer.py
VIEWER_CMD := cd $(STEPPING_DIR) && $(PY) urdf_viewer.py --urdf ../beta.urdf --fixed --yarp --left-port /urdf_viewer/left:o --right-port /urdf_viewer/right:o

.PHONY: help env check yarp gait sender viz viewer up clean deep-clean ports

env:
	@echo "export YARP_PORT=$(YARP_PORT)"

check:
	@command -v yarp >/dev/null || (echo "yarp not found in PATH" && exit 1)
	@command -v $(PY) >/dev/null || (echo "python3 not found in PATH" && exit 1)

yarp: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] YARP_PORT=$$YARP_PORT"; \
	$(YARP_SERVER)

gait: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] YARP_PORT=$$YARP_PORT"; \
	$(GAIT_CMD)

sender: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] YARP_PORT=$$YARP_PORT"; \
	$(SENDER_CMD)

viz: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] YARP_PORT=$$YARP_PORT"; \
	$(VIZ_CMD)

viewer: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] YARP_PORT=$$YARP_PORT"; \
	$(VIEWER_CMD)

up:
	@echo "Run in separate terminals:"
	@echo "  1) make yarp YARP_PORT=$(YARP_PORT)"
	@echo "  2) make gait YARP_PORT=$(YARP_PORT)"
	@echo "  3) make sender YARP_PORT=$(YARP_PORT)"
	@echo "  4) make viz YARP_PORT=$(YARP_PORT)"
	@echo "  5) make viewer YARP_PORT=$(YARP_PORT)"

ports: check
	@export YARP_PORT=$(YARP_PORT); \
	yarp name list || true

clean: check
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] killing processes..."; \
	pkill -f "yarpserver" || true; \
	pkill -f "gait.py" || true; \
	pkill -f "Sender.py" || true; \
	pkill -f "visulizer.py" || true; \
	pkill -f "urdf_viewer.py" || true; \
	sleep 1; \
	echo "[make] trying to unregister known ports..."; \
	for p in \
		/root fallback \
		/robot/cmd /debug/sender /debug/receiver \
		/gait/left/foot /gait/right/foot /gait/viz \
		/gait/left/angles /gait/right/angles \
		/viz/in \
		/urdf_viewer/left:o /urdf_viewer/right:o \
		/urdf_viewer/left_cmd:o /urdf_viewer/right_cmd:o \
		/urdf_viewer/left_cmd_v2:o /urdf_viewer/right_cmd_v2:o \
	; do \
		yarp clean $$p >/dev/null 2>&1 || true; \
	done; \
	echo "[make] done"

deep-clean: clean
	@export YARP_PORT=$(YARP_PORT); \
	echo "[make] yarp conf --clean"; \
	yarp conf --clean || true; \
	echo "[make] deep-clean done"
