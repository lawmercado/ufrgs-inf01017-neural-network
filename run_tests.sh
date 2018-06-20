python3 main.py --normalize --seed=5 test/wine/network1.txt test/wine/weights.txt sets/wine.txt > test/wine/test1.txt &
python3 main.py --normalize --seed=5 test/wine/network2.txt test/wine/weights.txt sets/wine.txt > test/wine/test2.txt &
python3 main.py --normalize --seed=5 test/wine/network3.txt test/wine/weights.txt sets/wine.txt > test/wine/test3.txt &
python3 main.py --normalize --seed=5 test/wine/network_lamda1.txt test/wine/weights.txt sets/wine.txt > test/wine/test4.txt &
python3 main.py --normalize --seed=5 test/wine/network_lamda2.txt test/wine/weights.txt sets/wine.txt > test/wine/test5.txt &
python3 main.py --normalize --seed=5 test/wine/network_lamda3.txt test/wine/weights.txt sets/wine.txt > test/wine/test6.txt &

python3 main.py --normalize --seed=5 test/diabetes/network1.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test1.txt &
python3 main.py --normalize --seed=5 test/diabetes/network2.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test2.txt &
python3 main.py --normalize --seed=5 test/diabetes/network3.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test3.txt &
python3 main.py --normalize --seed=5 test/diabetes/network_lamda1.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test4.txt &
python3 main.py --normalize --seed=5 test/diabetes/network_lamda2.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test5.txt &
python3 main.py --normalize --seed=5 test/diabetes/network_lamda3.txt test/diabetes/weights.txt sets/diabetes.txt > test/diabetes/test6.txt &

python3 main.py --normalize --seed=5 test/ionosphere/network1.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test1.txt &
python3 main.py --normalize --seed=5 test/ionosphere/network2.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test2.txt &
python3 main.py --normalize --seed=5 test/ionosphere/network3.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test3.txt &
python3 main.py --normalize --seed=5 test/ionosphere/network_lamda1.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test4.txt &
python3 main.py --normalize --seed=5 test/ionosphere/network_lamda2.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test5.txt &
python3 main.py --normalize --seed=5 test/ionosphere/network_lamda3.txt test/ionosphere/weights.txt sets/ionosphere.txt > test/ionosphere/test6.txt &

python3 main.py --normalize --seed=5 test/cancer/network1.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test1.txt &
python3 main.py --normalize --seed=5 test/cancer/network2.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test2.txt &
python3 main.py --normalize --seed=5 test/cancer/network3.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test3.txt &
python3 main.py --normalize --seed=5 test/cancer/network_lamda1.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test4.txt &
python3 main.py --normalize --seed=5 test/cancer/network_lamda2.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test5.txt &
python3 main.py --normalize --seed=5 test/cancer/network_lamda3.txt test/cancer/weights.txt sets/cancer.txt > test/cancer/test6.txt &
