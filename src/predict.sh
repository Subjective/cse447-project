#!/usr/bin/env bash
set -e
set -v
python src/lstm.py test --work_dir work --test_data $1 --test_output $2
