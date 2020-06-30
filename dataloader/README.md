MyCustomDataset:



ImageFolderDataLoader:
  using torchvision.datasets.ImageFolder API, a generic data loader where the images are arranged in this way:
      root/train/dog/xx1.png
	  root/train/dog/xx2.png
	  ...
	  root/val/cat/xx1.png
	  root/val/cat/xx2.png
	  ...
	  root/eval/people/xx1.png
	  root/eval/people/xx2.png
  
  input attribute:
      data_dir='./root'
	  partition=['train', 'val', 'eval']
  
  output attribute:
      dataloaders={'train':dataloader, 'val':dataloader, 'eval':dataloader}
	  dataset_sizes={'train':int, 'val':int, 'eval':int}
	  class_names=['label1', 'label2', ...]