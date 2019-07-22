# 3D-R2N2 Pytorch implementation

A simple implementation of Choy's work [3D-R2N2](<https://arxiv.org/abs/1604.00449>).

> * Prerequisite
>   > pip install -r requirements.txt
> * Download the dataset
>   > https://github.com/chrischoy/3D-R2N2
> * Split the dataset
>   > python split_dataset.py --root /your/dataset/path
> * Train and evaluate the model
>   > python main.py --your_configurations
> * Write the predictions to .binvox files
>   > python write_model.py --your_configurations

[Viewvox](https://www.patrickmin.com/viewvox/) under the directory ./objs visualizes the .binvox files.
