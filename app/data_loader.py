import os
import pandas as pd


# 프로젝트 루트 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# data 폴더 경로
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data(file_list):
    """
    여러 개의 엑셀 파일을 읽어서 하나의 DataFrame으로 반환
    """

    dfs = []

    for file_name in file_list:

        file_path = os.path.join(
            DATA_DIR,
            file_name
        )


        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"파일을 찾을 수 없습니다.\n{file_path}"
            )


        df = pd.read_excel(file_path)

        dfs.append(df)



    # 여러 데이터 합치기
    data = pd.concat(
        dfs,
        ignore_index=True
    )


    return data
