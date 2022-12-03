# ECE 591 (004) Fall 2022 project

## Background

Final project at [NC State University] [CSC 591/791 & ECE 591 (004), Fall 2022] course.

The goal of the project is to create a foundation for tracking algorithms fusion.

The project written in Python and implements the [Unscented Kalman Filter (UKF)] /
[Unscented Transformation (UT)] with minimal [external dependencies](/requirements.txt).

More information on UKF can be found under the [`./papers/`](/papers/) directory.

## Prerequisites

The minimum requirement to run the project is [attrs] (version 22.1.0) that can be installed using
the following command: `pip3 install "attrs>=22.1.0"`.

In addition to [attrs], [pytest], [PyHamcrest] and [Pylint] are used for testing, [Sphinx] is used
as the documentation ([docstrings]) generator and [Matplotlib] is used to display the results.

All of them can be installed using the following command:

```
pip3 install -r ./requirements.txt
```

## Run

To project entry point is `./src/main.py`, To run the project execute the following command:

```
python3 ./src/main.py
```

## Testing

To execute the tests ([pytest], [doctest] and [Pylint]), execute the following command:

```
sh ./src/run_tests.sh
```

## Documentation

To build the documentation HTML, execute the following command:

```
sh ./docs/build_docs.sh
```

And use your web browser to open the `index.html` file.

```
$ google-chrome ./docs/build/html/index.html
$ firefox ./docs/build/html/index.html
```

## Side note

As part of this project, several tools/tutorials were built for converting formats, e.g.

* GPX to XODR
* TIF to UMAP

More info can be found on a separate repository:
https://github.com/xqgex/SecurityofCyberPhysicalSystems_Formats

[attrs]: https://www.attrs.org/en/22.1.0/
[CSC 591/791 & ECE 591 (004), Fall 2022]: https://mankiyoon.github.io/courses/csc591_791/fa22/
[docstrings]: https://en.wikipedia.org/wiki/Docstring
[doctest]: https://docs.python.org/3/library/doctest.html
[Matplotlib]: https://matplotlib.org/3.6.0/tutorials/index.html
[NC State University]: https://wolfware.ncsu.edu/courses/details/?sis_id=SIS:2022:8:1:ECE:591:004
[PyHamcrest]: https://pyhamcrest.readthedocs.io/en/v2.0.4/
[Pylint]: https://pylint.pycqa.org/en/v2.15.7/
[pytest]: https://docs.pytest.org/en/7.2.x/
[Sphinx]: https://www.sphinx-doc.org/en/master/
[Unscented Kalman Filter (UKF)]: https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter
[Unscented Transformation (UT)]: https://en.wikipedia.org/wiki/Unscented_transform
