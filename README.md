Dataset and Network combinations:

--dataset cifar --model vgg16

OR

--dataset fashion_mnist --model cnn

<br>

To run our rate allocation, please install matlab.engine

To run our client selection methods in PB settings:
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --user_select cov
```

To run our client selection methods in PS settings:
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.0 --lr 0.01 --cov_analysis --user_select cov --shards_per_client 1
```

<br>

To compare with SOTA methods (follow above to change heterogeneity):
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --gpr
```
```shell
python main.py --dataset cifar --local_ep 3  --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --afl
```
```shell
python main.py --dataset cifar --local_ep 3  --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --powd
```

<br>

To run rate allocation:
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method binning  --simulate_quant
```
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method no_bin --simulate_quant
```
```shell
python main.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03  --cov_analysis --quant_method centralized --simulate_quant
```

<br>

To run client selection + rate allocation:
```shell
python main_both.py --dataset cifar --local_ep 3 --model vgg16 --epochs 20 --bias 0.25 --lr 0.03 --cov_analysis --quant_method binning --user_select cov --both
```
