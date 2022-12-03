#!/bin/bash
cd "$(dirname "$0")"

echo 'Run tests'
pytest --doctest-modules ./

python_code_files=$(find ./ -type f -iname '*.py' -and ! -iname '*_test.py')
echo "Run Python linter on:\n$python_code_files"
pylint --max-line-length=120 --max-args=6 --max-attributes=20 --disable=invalid-name $python_code_files

python_test_files=$(find ./ -type f -iname '*_test.py')
echo "Run Python linter on:\n$python_test_files"
pylint --max-line-length=120 --disable=missing-class-docstring --disable=missing-function-docstring $python_test_files
