from torchvision import transforms
import torchvision
from torch.utils import data
def load_data_MINST(batch_size,resize=None, thread=4):
    """input batch_size,"""
    """下载fashion-mnist数据集，并将其加载到内存中"""
    trans = [transforms.ToTensor()]
    """生成对象"""
    """将读进来的数据从[0,255]转到[0,1]"""
    if resize:
        """如果resize那为1"""
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    """注意，这个是加载到内存里"""

    mnist_train = torchvision.datasets.MNIST(
        root="../data_MINST", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.MNIST(
        root="../data_MINST", train=False, transform=trans, download=False)

    return (
        data.DataLoader(mnist_train, batch_size,
                        num_workers=thread),
        data.DataLoader(mnist_test, batch_size,
                        num_workers=thread)
    )
if __name__ == '__main__':
    iter_test,iter_train=load_data_MINST(20,32)
    for x,y in iter_test:
        print(x.shape)
        break