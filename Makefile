.PHONY: pcl_helper build build_inplace doc develop install uninstall clean

all: build

pcl_helper:
	python setup.py build_pcl_helper

build:
	python setup.py build

build_inplace:
	python setup.py build --inplace

doc: build_inplace
	python setup.py build_sphinx

develop:
	pip install -e .

install:
	pip install .

uninstall:
	pip uninstall pydriver

clean:
	python setup.py clean
