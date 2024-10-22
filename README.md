# MLGNN

MLGNN is a optimization-based meta-learning method designed typically designed for graph neural network for few-shot learning.
This meta-learning method is a task set-based method which not only devide the dataset into tasks but also construct task sets for efficient tarining.
This code is constructed based on Graph Neural Network For Few-shot Learning, whose pytorch implementation is https://github.com/ylsung/gnn_few_shot_cifar100, which in our paper serves as the baseline.

## Usage

To train and test the model, run `main.py`.

## Dependencies
- Python 3.8
- PyTorch 2.2.0

## Dataset

The dataset should be organized in the `data/` directory with the following structure:

```
data/
├── train/
└── test/
```

## Model

The model is defined in `models.py` and consists of convolutional layers followed by graph convolutional layers.

## License

This project is licensed under the MIT License.

## Citation
please kindly cite this paper if our HRRPGraphNet can give you any inspiration for your research, thanks a lot.

## Contact

Lingfeng Chen

Email: chenlingfeng@nudt.edu.cn
