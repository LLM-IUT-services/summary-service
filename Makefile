# Define environment variables
export HUGGINGFACE_API_KEY=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

# Define Python version and virtual environment directory
PYTHON=python3.12
VENV_DIR=venv

# Install dependencies
install:
	@echo "[+] Checking if $(PYTHON) is installed..."
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "[-] Error: $(PYTHON) is not available. Please install Python 3.12."; exit 1; }
	@echo "[+] Creating virtual environment with $(PYTHON)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "[+] Activating virtual environment and installing requirements..."
	@$(VENV_DIR)/bin/pip install --upgrade pip
	@$(VENV_DIR)/bin/pip install -r requirements.txt
	@mkdir -p models/fa 
	@mkdir -p models/en 
	@wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip  
	@wget https://alphacephei.com/vosk/models/vosk-model-small-fa-0.42.zip 
	@unzip vosk-model-small-fa-0.42.zip -d models/fa/
	@unzip vosk-model-small-en-us-0.15.zip -d models/en/
	@rm  vosk-model-small-fa-0.42.zip vosk-model-small-en-us-0.15.zip
	@$(VENV_DIR)/bin/huggingface-cli login --token $(HUGGINGFACE_API_KEY)

run:
	@echo "[+] Running the application..."
	$(VENV_DIR)/bin/python app.py

run-threaded:
	@echo "[+] Running the application with threads ..."
	@venv/bin/gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 app:app

.PHONY: test
# Run tests with pytest
test:
	@echo "[+] Running tests..."
	PYTHONPATH=. $(VENV_DIR)/bin/pytest

clean:
	@echo "[+] Cleaning up..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf .pytest_cache

help:
	@echo "Available commands:"
	@echo "  make run-threaded - Run but with threads"
	@echo "  make install      - Install dependencies and setup virtual environment using Python 3.12"
	@echo "  make run          - Run the Flask application"
	@echo "  make test         - Run tests with pytest"
	@echo "  make clean        - Remove virtual environment and cache files"
