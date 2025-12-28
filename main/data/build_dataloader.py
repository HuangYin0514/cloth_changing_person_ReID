import torchvision.transforms as T
from torch.utils.data import DataLoader

from .datasets import LTCC
from .image_dataset import ImageDataset
from .image_dataset_img_mask import ImageDataset_Img_Mask
from .image_transforms import RandomErasing
from .samplers import RandomIdentitySampler_cc


def build_img_transforms(config):
    transform_train = T.Compose(
        [
            T.Resize((config.DATA.IMAGE_HEIGHT, config.DATA.IMAGE_WIDTH)),
            T.RandomHorizontalFlip(p=0.5),
            T.Pad(padding=10),
            T.RandomCrop((config.DATA.IMAGE_HEIGHT, config.DATA.IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0]),
        ]
    )
    transform_test = T.Compose(
        [
            T.Resize((config.DATA.IMAGE_HEIGHT, config.DATA.IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return transform_train, transform_test


def build_dataloader(config):

    # Transforms
    transform_train, transform_test = build_img_transforms(config)

    # Dataloader
    if config.DATA.TRAIN_DATASET == "ltcc":
        dataset = LTCC(root=config.DATA.TRAIN_ROOT)

        train_dataset = ImageDataset_Img_Mask(dataset.train, transform=transform_train)
        query_dataset = ImageDataset(dataset.query, transform=transform_test)
        gallery_dataset = ImageDataset(dataset.gallery, transform=transform_test)

        train_sampler = RandomIdentitySampler_cc(dataset.train, batch_size=config.DATA.BATCHSIZE, num_instances=config.DATA.NUM_INSTANCES)
        train_loader = DataLoader(
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=config.DATA.BATCHSIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        query_loader = DataLoader(
            dataset=query_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        gallery_loader = DataLoader(
            dataset=gallery_dataset,
            batch_size=config.DATA.TEST_BATCH,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

        return dataset, train_loader, query_loader, gallery_loader
