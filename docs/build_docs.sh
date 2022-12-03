#!/bin/bash
cd "$(dirname "$0")"

sphinx-apidoc --force -o ./source/ ../src/
make clean
make html
