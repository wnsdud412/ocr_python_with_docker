import cv2
import os
import pandas as pd
import json
import re
from korean_ocr.layers.text import DEFAULT_SPECIAL_CHARACTERS


class OCRDataset:
    """
    READER Class  FOR OCR Dataset

    :param label_df: read_label_dataframe()으로부터 읽어들인 label과 이미지 경로 정보가 담긴 pandas.DataFrame
    :param height : 이미지의 높이, OCRDataset은 항상 같은 높이를 가진 이미지를 Return
    :return:
        OCRDataset Class.

    :examples:
        >>> label_df = read_label_dataframe(os.path.join(HANDWRITTEN_DIR, 'dataset_info.json'))
        >>> dataset = OCRDataset(label_df) # 데이터 셋 선언하기
        >>> len(dataset) # 데이터 셋의 갯수 구하기
        >>> dataset[3] # 데이터 셋 내 3번째 이미지 파일과 라벨 가져오기
        >>> dataset[[3,5,10]] # "" [3,5,10]번째 이미지 파일과 라벨 가져오기
        >>> dataset[2:4] # "" [2,4)번째 이미지 파일과 라벨 가져오기
    """
    def __init__(self, label_df:pd.DataFrame, height=64):
        assert len({'text', 'file_path'} - set(label_df.columns)) == 0, (
            "label_df는 text와 file_path 두 column을 필수로 가지고 있어야 합니다.")
        self.label_df = label_df
        self.height = height
        # index를 0부터 len(label_df)로 재배치
        self.label_df = self.label_df.reindex()

    def __len__(self):
        """
        데이터 셋 내 파일의 갯수

        :return:
        """
        return len(self.label_df)

    def __getitem__(self, index):
        if isinstance(index, int):
            target = self.label_df.iloc[index]
            word = target.text
            image = self.read_image(target.file_path)
            return image, word
        else:
            targets = self.label_df.iloc[index]
            images = []; words = []
            for idx, target in targets.iterrows():
                images.append(self.read_image(target.file_path))
                words.append(target.text)
            return images, words

    def read_image(self, file_path):
        """
        Gray Scale로 파일 읽어오기

        :param file_path: 이미지 파일 경로
        :return:
        """
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        assert image is not None, f"경로({file_path})에 이미지가 존재하지 않습니다."
        height, width = image.shape
        new_height = int(self.height)
        new_width = int(self.height / height * width)
        image = cv2.resize(image, (new_width, new_height))
        return image

    def shuffle(self):
        """
        데이터 셋을 섞음

        :return:
        """
        # index 순서를 재배치(섞음)
        self.label_df = self.label_df.sample(frac=1).reset_index(drop=True)

    def get_config(self):
        return {
            "name": "OCRDataset",
            "label_df": self.label_df,
            "height": self.height
        }


def read_label_dataframe(info_path: str):
    """
    dataset_info.json으로부터,
    파일경로 별 Label 정보를 가져오는 메소드

    :param info_path : 읽고자 하는 dataset_info.json의 경로

    :return:
        pandas.Dataframe
            |-file_path : 이미지 파일 경로
            |-text      : 해당 이미지의 글자
            |-width     : 해당 이미지의 폭 길이
            |-height    : 해당 이미지의 높이 길이
    """
    # json 파일 읽어오기
    with open(info_path, 'r') as f:
        info = json.load(f)

    # annotation 정보 가져오기
    image_df = pd.DataFrame(info['images'])
    image_df = image_df[['id', 'file_name', 'width', 'height']]
    anno_df = pd.DataFrame(info['annotations'])
    anno_df = anno_df[['id', 'text']]
    if 'type' in info['annotations'][0]['attributes']:
        type_exist = True
        anno_df.loc[:, "type"] = [row['attributes']['type']
                                  for row in info['annotations']]
    else:
        type_exist = False
    label_df = pd.merge(image_df, anno_df, on='id')

    # 파일이름을 파일경로로 변경
    data_dir = os.path.dirname(os.path.abspath(info_path))
    image_dir = os.path.join(data_dir, 'images')
    label_df['file_path'] = (
        label_df['file_name'].apply(lambda x: os.path.join(image_dir, x)))
    if type_exist:
        label_df = label_df[['file_name', 'file_path',
                             'text', 'width', 'height', 'type']]
    else:
        label_df = label_df[['file_name', 'file_path',
                             'text', 'width', 'height']]
    return label_df


def filter_out_dataframe(df,
                         data_types=('글자(음절)','단어(어절)','문장'),
                         special_characters=DEFAULT_SPECIAL_CHARACTERS,
                         max_text_length=None):
    """
    pd.DataFrame에서 학습에서 불필요한 케이스들을 제거하는 메소드

    :param df: read_label_dataframe으로 읽어들인 pd.DataFrame
    :param data_types: data_types[ '글자(음절)', '단어(어절)', '문장'] 중 학습에 사용할 데이터 종류 집합들
    :param special_characters: 포함시킬 특수문자 집합들
    :param max_text_length:
    :return:
    """
    if 'type' in df.columns and data_types is not None:
        df = df[df.type.isin(data_types)]

    # 한글은 필수로 포함
    valid_character_pattern = "^[가-힣"
    for char in special_characters:
        if char.isalnum():
            # 영어와 숫자의 경우
            valid_character_pattern = valid_character_pattern + "|" + char
        elif char == ' ':
            valid_character_pattern = valid_character_pattern + "|\s"
        else:
            # 그외 특수문자의 경우
            valid_character_pattern = valid_character_pattern + "|\\" + char
    valid_character_pattern = valid_character_pattern + ']+$'

    re_valid_characters = re.compile(valid_character_pattern)

    df = df[df.text.apply(
        lambda x: bool(re_valid_characters.fullmatch(x)))]

    if max_text_length is not None:
        df = df[df.text.apply(lambda x: len(x) <= max_text_length)]

    return df