from tensorflow.python.keras.utils import Sequence
import numpy as np
import imgaug.augmenters as iaa

DEFAULT_AUGMENTS = {
    "blur": 3.,
    "gaussian_noise": 5.,
    "multiply": .2,
    "scale": .2,
    "rotate": 5,
    "shear": 5.,
    "elastic": .5
}


class DataGenerator(Sequence):
    "DataSet의 값을 배치 단위로 Return"

    def __init__(self,
                 dataset,
                 batch_size=128,
                 mask_value=-1,
                 shuffle=True,
                 augments=DEFAULT_AUGMENTS):
        """
        배치 단위로 데이터를 쪼개고, 학습할 수 있는 형태로 라벨을 변형시키는 클래스

        param
        :param dataset : instance of class 'Dataset'
        :param batch_size : the number of batch
        :param shuffle : whether shuffle dataset or not
        :param augments : dict,
            :key blur: 가우시안 블러 효과, 커질수록 노이즈의 효과가 두드러짐
            :key noise: 가우시안 노이즈
            :key multiply: 이미지 strength 효과
            :key scale: 이미지 확대 및 축소
            :key rotate: 회전의 범위, ex 5인 경우, (-5도 ~ 5도 사이 회전)
            :key shear: 전단 회전 효과
            :key elastic: elastic transform 효과

        """
        self.dataset = dataset
        if isinstance(augments, dict):
            augments['height'] = self.dataset.height
            self.augment_pipeline = create_augmentation_pipeline(**augments)
        else:
            self.augment_pipeline = None
        self.batch_size = batch_size
        self.mask_value = mask_value
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.dataset) // self.batch_size

    def __getitem__(self, index):
        "Generator one batch of dataset"
        images, labels = self.dataset[self.batch_size * index:
                                      self.batch_size * (index + 1)]

        if self.augment_pipeline is not None:
            images = self.augment_pipeline.augment_images(images)

        max_height = self.dataset.height
        max_width = np.max([image.shape for image in images], axis=0)[1]
        batch_images = np.zeros((len(images), max_height, max_width))
        max_len = max([len(label) for label in labels]) + 1
        unicode_arr = np.ones((self.batch_size, max_len), dtype=np.int) * self.mask_value
        unicode_arr = unicode_arr.astype(np.int)

        for idx, (image, label) in enumerate(zip(images, labels)):
            unicode_arr[idx, :len(label)] = np.array([ord(char) for char in label])
            unicode_arr[idx, len(label)] = ord('\n')
            batch_images[idx, :image.shape[0], :image.shape[1]] = image[:max_height, :max_width]
        decoder_inputs = np.roll(unicode_arr, 1, axis=1)
        decoder_inputs[:, 0] = ord('\n')
        X = {
            "images": batch_images,
            "decoder_inputs": decoder_inputs,
        }
        return X, unicode_arr

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        if self.shuffle:
            self.dataset.shuffle()


def create_augmentation_pipeline(
    height=64, blur=3., gaussian_noise=5.,
    multiply=.2, scale=.2, rotate=5, shear=5., elastic=.5):
    """
    이미지 증강 Function

    imgaug을 기본으로 Wrapping한 Function으로,
    Text Recognition Dataset에 자주 쓰이는 Data Augmentation
    기법들을 추려 설정해놓았습니다.

    :param blur: 가우시안 블러 효과, 커질수록 노이즈의 효과가 두드러짐
    :param noise: 가우시안 노이즈
    :param multiply: 이미지 strength 효과
    :param scale: 이미지 확대 및 축소
    :param rotate: 회전의 범위, ex 5인 경우, (-5도 ~ 5도 사이 회전)
    :param shear: 전단 회전 효과
    :param elastic: elastic transform 효과

    """
    blur_range = (0., blur)
    noise_range = (0., gaussian_noise)
    multiply_range = (1. - multiply, 1. + multiply)
    scale_range = (1. - scale, 1. + scale)
    rotate_range = (-rotate, rotate)
    shear_range = (-shear, shear)
    elastic_range = (0., elastic)

    return iaa.Sequential([
        iaa.GaussianBlur(sigma=blur_range),
        iaa.AdditiveGaussianNoise(loc=0,
                                  scale=noise_range,
                                  per_channel=False),
        iaa.Multiply(multiply_range),
        iaa.Affine(scale=scale_range,
                   rotate=rotate_range,
                   shear=shear_range,
                   fit_output=True),
        iaa.ElasticTransformation(alpha=elastic_range),
        iaa.CropToFixedSize(height*100, height)
    ])