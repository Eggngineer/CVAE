import torch, torchvision
import torchvision.transforms as transforms

# def : DataLoader 
def  MNIST_Loader(conf):
  train_dataset = torchvision.datasets.MNIST(
      root = conf['roots']
      , train = conf['train']
      , transform = transforms.ToTensor()
      , download = conf['downloads']
  )

  train_dataloader = torch.utils.data.DataLoader(
      dataset = train_dataset
      , batch_size = conf['batch_size']
      , shuffle = conf['shuffle']
      , num_workers = 0
  )
  return train_dataloader 