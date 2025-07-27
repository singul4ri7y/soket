from __future__ import annotations
from typing import Type, Tuple
from soket import Tensor, exp
from soket.transforms import ToTensor
from soket.optim import Adam, Optimizer
from soket.utils.data import DataLoader
from soket.utils.data.datasets import MNIST
import soket.nn as nn
import soket.nn.init as F


# Loss criterion being used.
criterion = nn.SoftmaxCrossEntropyLoss()


class ResidualBlock(nn.Sequential):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 100,
        norm: nn.Module = nn.LayerNorm,
        drop_prob: float = 0.01
    ):
        fn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=drop_prob),
            nn.Linear(hidden_dim, hidden_dim),
            norm(hidden_dim)
        )

        super().__init__(
            nn.Residual(fn),
            nn.ReLU()
        )


class MLPResNet(nn.Sequential):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 100,
        num_blocks: int = 3,
        num_classes: int = 10,
        norm: nn.Module = nn.LayerNorm,
        drop_prob: float = 0.01
    ):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),

            *[ResidualBlock(hidden_dim, hidden_dim, norm=norm, drop_prob=drop_prob)
              for _ in range(num_blocks)],
            
            nn.Linear(hidden_dim, num_classes)
        )


def mlp_resnet_get_accuracy(Z: Tensor, y: Tensor) -> float:
    """ A very crude implementation of model accuracy. """

    # Apply softmax on logits then find and compare argmax to 
    # calculate the accuracy.
    # Warning: Soket Tensor does not support max() operation yet, hence
    # calculating softmax in this way might get unstable.
    e = exp(Z)
    softmax = e / e.sum(-1, keepdims=True)

    return (softmax.argmax(-1) == y).mean().item()


def mlp_resnet_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optim: Optimizer = None
) -> Tuple[float]:
    total_loss, total_acc = 0.0, 0.0

    for X, y in dataloader:
        logits = model(X)
        loss = criterion(logits, y)

        # Setting `optim` would suggest the model is being trained. Hence, calculate
        # gradients and optimize the model.
        if optim is not None:
            loss.backward()
            optim.step()

        total_loss += loss.item()
        total_acc += mlp_resnet_get_accuracy(logits, y)
    
    total_loss /= dataloader.max_iter
    total_acc /= dataloader.max_iter

    return total_loss, total_acc


def train_and_test_mlp_resnet_with_mnist(
    batch_size: int = 100,
    epochs: int = 10,
    optimizer: Type[Optimizer] = Adam,
    lr: float = 0.01,
    weight_decay: float = 0.001,
    hidden_dim: int = 100,
    data_dir: str = 'data'
):
    # Load MNIST train and test datasets and initialize loaders.
    train_loader = DataLoader(
        MNIST(
            f'{data_dir}/train-images-idx3-ubyte.gz',
            f'{data_dir}/train-labels-idx1-ubyte.gz',
            transforms=ToTensor(),
            target_transforms=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        MNIST(
            f'{data_dir}/t10k-images-idx3-ubyte.gz',
            f'{data_dir}/t10k-labels-idx1-ubyte.gz',
            transforms=ToTensor(),
            target_transforms=ToTensor()
        ),
        batch_size=batch_size,
        shuffle=False
    )

    model = MLPResNet(28 * 28, hidden_dim=hidden_dim, num_classes=10)

    # Kaiming initialization of the model parameters.
    for m in model.modules():
        if isinstance(m, nn.Linear):
            F.kaiming_normal(m.weight)

    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for e in range(epochs):
        train_loss, train_acc = mlp_resnet_epoch(model, train_loader, optim)
        print(f'Epoch: {e}, train loss: {train_loss}, train acc: {train_acc * 100} %')
    
    model.eval()
    test_loss, test_acc = mlp_resnet_epoch(model, test_loader)
    print(f'Test loss: {test_loss}, test acc: {test_acc * 100} %')

if __name__ == '__main__':
    train_and_test_mlp_resnet_with_mnist(
        batch_size=250,
        epochs=10,
        lr=0.005,
        weight_decay=0.01,
        data_dir='data'
    )