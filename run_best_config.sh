python3 main.py --normalize --seed=5 test/wine/best_config.txt test/wine/weights.txt sets/wine.txt > test/wine/best_config_result.txt &

python3 main.py --normalize --seed=5 test/diabetes/best_config.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/best_config_result.txt &

python3 main.py --normalize --seed=5 test/ionosphere/best_config.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/best_config_result.txt &

python3 main.py --normalize --seed=5 test/cancer/best_config.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/best_config_result.txt &
