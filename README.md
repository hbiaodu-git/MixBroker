# Breaking the Anonymity of Ethereum Mixing Services Using Graph Feature Learning

This is a Pytorch implementation of MixBroker, as described in the following:
> Breaking the Anonymity of Ethereum Mixing Services Using Graph Feature Learning


## Requirements
For hardware configuration, the experiments are conducted at Windows 10 with the Intel Core i7-9750H Six-Core CPU @ 2.60GHz, NVIDIA GeForce RTX 2070 GPU, and 16GB RAM.
For software configuration, all model are implemented in
- Python 3.7
- Pytorch 1.9.1
- CUDA 10.1
- Numpy 1.21.6
- Pandas 0.25.1
- Pytorch-Geometric 2.1.0
- Scikit-learn 1.0.2

## Data

We store the acquired Tornado Cash data as well as the training data for subsequent use in the 'Dataset/' folder.
There are five subfolders under the 'Dataset/' folder:

- External: It contains the external transaction data of Ethereum using Tornado Cash acquired from [Etherscan](https://etherscan.io/).
- Internal: It contains the internal transaction data of Ethereum using Tornado Cash acquired from Etherscan.
- Concat: It contains the complete transactions of the four mixing pools of Tornado Cash as well as the Relayer data.
- Ens: It contains the Ethereum Name Service (ENS) data used for experiments.
- Graph: It contains the graph node feature data and the training data.

## How to Run

There are several data paths you should change in the code.

### MixBroker

### Data Preparation

Here are three steps to prepare the ground-truth dataset. 

#### Step 1: Prepare Tornado Cash transaction dataset
```commandline
python TC_data_prepare.py
```

#### Step 2: Prepare ENS address pair dataset
```commandline
python ENS_data_prepare.py
```

#### Step 3: Generate ground-truth dataset
```commandline
python ground_truth_generate.py
```

### Feature Extraction

The following command can extract features on the target account nodes in MIG to obtain more dimensional
node feature information.
```commandline
python feature_extract.py
```

### Model Training and Testing

If you want to train the model with 10-fold validation, you can use this command.
```commandline
python main.py
```

## Citation

If you find this work useful, please cite the following:
```article
H. Du, Z. Che, M. Shen, L. Zhu and J. Hu, "Breaking the Anonymity of Ethereum Mixing Services Using Graph Feature Learning," 
in IEEE Transactions on Information Forensics and Security, vol. 19, pp. 616-631, 2024, doi: 10.1109/TIFS.2023.3326984.
```

The BibTeX is as the following:
```bib
@ARTICLE{10292691,
  author={Du, Hanbiao and Che, Zheng and Shen, Meng and Zhu, Liehuang and Hu, Jiankun},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Breaking the Anonymity of Ethereum Mixing Services Using Graph Feature Learning}, 
  year={2024},
  volume={19},
  number={},
  pages={616-631},
  doi={10.1109/TIFS.2023.3326984}
}
```

## Q&A

If you have any questions, you can contact me (duhanbiao@bit.edu.cn), and I will reply as soon as I see the email.