#!/usr/bin/env bash
set -e
set -v
python src/logistic_regression.py test --work_dir work --test_data $1 --test_output $2
