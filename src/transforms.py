import torch

class ToTensor:
    def __call__(self, sample):
        bucket, image = sample['bucket'], sample['image']
        return {
            'bucket': torch.from_numpy(bucket),
            'image': torch.from_numpy(image)
        }