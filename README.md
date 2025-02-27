## Simple Usage

### Dataset
- `mydataset/data_process_my.mlx`: Data preprocessing code (placed in the `script` directory of [sonic160/dtr_digital_model_simulink](https://github.com/sonic160/dtr_digital_model_simulink), used with Matlab for preprocessing, **not necessary if you only need code from our paper**)
- `mydataset/my_dataset_train.mat`: Training set (will be split into train and validation sets in the Jupyter file)
- `mydataset/my_dataset_test.mat`: Test set

### Scripts
- `scripts/models.py`: All model definitions used in the paper
- `scripts/train_ai_pytorch_DANN.ipynb`: Train and test using DANN, **run this file**
- `scripts/train_ai_pytorch_main.ipynb`: Train and test using models other than DANN, **run this file**

---

## 使用

### Dataset
- `mydataset/data_process_my.mlx`: 数据预处理代码（放置在 [sonic160/dtr_digital_model_simulink](https://github.com/sonic160/dtr_digital_model_simulink) 的 `script` 目录下，使用 Matlab 进行预处理，**如果只需要本论文的内容，实际不需要考虑**）
- `mydataset/my_dataset_train.mat`: 训练集（将在 Jupyter 文件中划分为训练集和验证集）
- `mydataset/my_dataset_test.mat`: 测试集

### Scripts
- `scripts/models.py`: 论文中使用的所有模型定义
- `scripts/train_ai_pytorch_DANN.ipynb`: 使用 DANN 进行训练和测试，**运行此文件**
- `scripts/train_ai_pytorch_main.ipynb`: 使用除 DANN 之外的其他模型进行训练和测试，**运行此文件**

