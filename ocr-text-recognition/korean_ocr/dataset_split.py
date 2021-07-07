"""
아래 스크립트에 학습 / 검증 / 평가 데이터셋을 나누는 과정을 기술하였습니다.
"""

from korean_ocr.data.dataset import read_label_dataframe
from korean_ocr.data.dataset import filter_out_dataframe
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import json


ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'datasets')


if __name__ == "__main__":
    handwritten_df = read_label_dataframe(
        os.path.join(DATA_DIR, "handwritten/dataset_info.json"))
    printed_df = read_label_dataframe(
        os.path.join(DATA_DIR, "printed/dataset_info.json"))

    handwritten_df.loc[:, "ref"] = 'handwritten'
    printed_df.loc[:, "ref"] = "printed"

    total_df = pd.concat([handwritten_df, printed_df])

    filtered_df = filter_out_dataframe(total_df)  # 영어 철자가 포함된 데이터 제거

    train_dfs = []
    test_dfs = []
    valid_dfs = []

    """
    글자(음절) 데이터
    
        글자(음절) 당 이미지를 동일 비율(7:1:2)로 나누어 따로 구성
        => train, test, validation에는 모든 글자 이미지를 포괄할 수 있도록 함
    """

    char_df = filtered_df[filtered_df.type == '글자(음절)']
    for i, sub_df in char_df.groupby(['text', 'ref']):
        train_sub, test_sub = (
            train_test_split(sub_df, test_size=.2, random_state=10))
        train_sub, valid_sub = (
            train_test_split(train_sub, test_size=.125, random_state=10))

        train_dfs.append(train_sub)
        test_dfs.append(test_sub)
        valid_dfs.append(valid_sub)

    """
    단어(어절) 데이터
    
        단어 별로 나누어서 따로 학습
        => train, test, validation에는 각기 다른 단어로 적힌 이미지를 포함함
    """

    word_df = filtered_df[filtered_df.type == '단어(어절)']
    unique_words = word_df.text.unique()
    train_words, test_words = (
        train_test_split(unique_words, test_size=.2, random_state=10))
    train_words, valid_words = (
        train_test_split(train_words, test_size=.125, random_state=10))

    train_sub = word_df[word_df.text.isin(train_words)]
    valid_sub = word_df[word_df.text.isin(valid_words)]
    test_sub = word_df[word_df.text.isin(test_words)]

    train_dfs.append(train_sub)
    valid_dfs.append(valid_sub)
    test_dfs.append(test_sub)

    """
    문장 데이터
    
        문장 별로 나누어서 따로 학습
        => train, test, validation에는 각기 다른 단어로 적힌 이미지를 포함함
    """

    sent_df = filtered_df[filtered_df.type == '문장']
    unique_sents = sent_df.text.unique()
    train_sents, test_sents = (
        train_test_split(unique_sents, test_size=.2, random_state=10))
    train_sents, valid_sents = (
        train_test_split(train_sents, test_size=.125, random_state=10))

    train_sub = sent_df[sent_df.text.isin(train_sents)]
    valid_sub = sent_df[sent_df.text.isin(valid_sents)]
    test_sub = sent_df[sent_df.text.isin(test_sents)]

    train_dfs.append(train_sub)
    valid_dfs.append(valid_sub)
    test_dfs.append(test_sub)

    train_df = pd.concat(train_dfs)
    valid_df = pd.concat(valid_dfs)
    test_df = pd.concat(test_dfs)

    train_df.loc[:, "set_type"] = 'train'
    valid_df.loc[:, "set_type"] = 'validation'
    test_df.loc[:, "set_type"] = 'test'
    all_df = pd.concat([train_df, valid_df, test_df])

    """
    Train / Test / Validation 나눈 결과를 저장하는 공간
    """
    for set_type in ['train', 'validation','test']:
        sub_df = all_df[all_df.set_type == set_type]
        for ref in ['printed', 'handwritten']:
            values = sub_df.loc[
                sub_df.reference == ref, 'file_name'].values.tolist()
            with open(os.path.join(DATA_DIR, f"{ref}/train.json"), 'w') as f:
                json.dump(values, f)

