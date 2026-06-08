# FireGuard IoT Makefile

.PHONY: install run-dashboard run-detector test clean

# Virtual environment setup
install:
	python -m venv venv
	./venv/Scripts/pip install --upgrade pip
	./venv/Scripts/pip install -r requirements.txt

# Run Streamlit Web Dashboard
run-dashboard:
	./venv/Scripts/streamlit run dashboard.py

# Run Standalone OpenCV Webcam detector
run-detector:
	./venv/Scripts/python fire_detection.py

# Run unit tests
test:
	./venv/Scripts/pytest -v

# Clean temporary cache files
clean:
	rm -rf __pycache__ .pytest_cache tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
