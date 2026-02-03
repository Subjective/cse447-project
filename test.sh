python src/lstm.py train --work_dir work
python src/lstm.py test --work_dir work --test_data example/input.txt --test_output pred.txt
python grader/grade.py pred.txt example/answer.txt --verbose
