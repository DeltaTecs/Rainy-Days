.PHONY: env test unittest

env:
	conda env create -f environment.yml || conda env update -f environment.yml

test:
	PYTHONPATH=$$(pwd) conda run -n modis-ml pytest

unittest:
	PYTHONPATH=$$(pwd) conda run -n modis-ml python -m unittest discover -v
