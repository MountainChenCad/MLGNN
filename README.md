# MLGNN

MLGNN (Meta-Learning for Graph Neural Network) is a optimization-based meta-learning method designed typically designed for graph neural network for few-shot learning.
This meta-learning method is a task set-based method which not only devide the dataset into tasks but also construct task sets for efficient tarining.
This code is constructed based on Graph Neural Network For Few-shot Learning, whose pytorch implementation is https://github.com/ylsung/gnn_few_shot_cifar100, which in our paper serves as the baseline.

## Usage

To train and test the model, run `main.py`.
To set the training hyper-parameters turn to `argument.py`.
Here, you may choose the dataset to be either `gaf12` or `hrrp3`, which will lead to dataloader of gaf/hrrp respectively.
Furthermore, if you choose `gaf12`, then in `trainer.py` you change the code in class `gnnModel` as:
```
        self.cnn_feature = EmbeddingCNN2D(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
```
On the other hand, if you choose `hrrp3`, then you should change the code to:
```
        self.cnn_feature = EmbeddingCNN1D(image_size, cnn_feature_size, cnn_hidden_dim, cnn_num_layers)
```
That is because hrrp is 1-D and gaf is 2-D, thereby different embedding network is needed.

## Dependencies
- Python 3.8
- PyTorch 2.2.0

## Dataset

The dataset should be organized in the `train/` directory with the following structure:

```
data/
├── hrrp3/
  └── train/
└── gaf12/
  └── train/
```

## Model

The model is defined in `trainer.py` and `gnn.py`.

## License

This project is licensed under the MIT License.

## Citation
please kindly cite this paper if our MLGNN can give you any inspiration for your research, thanks a lot.

## Contact

Lingfeng Chen

Email: chenlingfeng@nudt.edu.cn
