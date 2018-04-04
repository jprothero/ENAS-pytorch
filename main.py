from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import FLAGS
from lib.train import train
from lib.test import test
from lib.controller import Controller
from lib.child import make_child

def main():
    controller = Controller()

    kwargs = {"num_workers": 1, "pin_memory": True} if FLAGS.CUDA else {}
    train_loader = DataLoader(datasets.Omniglot(
        "data", train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std) should add for better performance, + other transforms
        ])), batch_sizer=FLAGS.BATCH_SIZE, shuffle=True, **kwargs)

    test_loader = DataLoader(datasets.Omniglot(
        "data", train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean, std) should add for better performance, + other transforms
        ])), batch_sizer=FLAGS.BATCH_SIZE, shuffle=True, **kwargs)

    if FLAGS.TRAIN:
        train(controller, train_loader)
    else:
        test(controller, test_loader)

if __name__ == "__main__":
    main()
