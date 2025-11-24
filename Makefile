PYTHON=python
PACKAGE=microseg

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check src tests
	black --check src tests || (echo "Run 'black src tests' to auto-format"; exit 1)

format:
	black src tests

test:
	pytest -q

train:
	$(PYTHON) -m microseg.train --config configs/unet_example.yaml

infer:
	$(PYTHON) -m microseg.infer \
		--config configs/unet_example.yaml \
		--checkpoint runs/example/checkpoints/best.pt \
		--input_folder data/val/images \
		--output_folder predictions

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ coverage.xml