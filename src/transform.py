from torchvision import transforms


def transform(train=False, normalize=True, augment=False):
    transform_list = []
    transform_list.append(transforms.ToPILImage())
    transform_list.append(transforms.Resize((32, 32)))

    if train and augment:
        transform_list.extend(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )

    transform_list.append(transforms.ToTensor())

    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)
