# Define environment variables
export HUGGINGFACE_API_KEY=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

# Define virtual environment
VENV_DIR=venv

# Install dependencies
install:
	@echo "[+] Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "[+] Activating virtual environment and installing requirements..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

run:
	@echo "[+] Running the application..."
	$(VENV_DIR)/bin/python app.py

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
	@echo "  make install   - Install dependencies and setup virtual environment"
	@echo "  make run       - Run the Flask application"
	@echo "  make test      - Run tests with pytest"
	@echo "  make clean     - Remove virtual environment and cache files"
