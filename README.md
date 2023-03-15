# GNPD
Code for Graph Non-Parametric Diffusion (GNPD) based on the TMLR paper "Parameter Efficient Node Classification on Homophilic Graphs"


* **LightGBM version of our method**

```bash
python main.py --model linear --dropout 0.5 --epochs 500
python main.py --model linear --pretrain --correction-alpha 0.99 --smoothing-alpha 0.75 --correction-adj DA --autoscale
```


* **Linear version of our method**

```bash
python main.py --model linear --dropout 0 --epochs 60
python main.py --model linear --pretrain --correction-alpha 1 --smoothing-alpha 0.92 --autoscale --aggregator Linear
```

This code is based on the [DGL implementation](https://github.com/dmlc/dgl/blob/master/examples/pytorch/correct_and_smooth/README.md) of [Correct&Smooth](https://github.com/CUAI/CorrectAndSmooth). 
