import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import os


def run_analysis(file_list):

    # =========================
    # 데이터 불러오기
    # =========================

    dfs = []

    for file in file_list:

        df = pd.read_excel(file)

        # 컬럼명 지정
        df.columns = [
            "지역",
            "X1",
            "X2",
            "X3",
            "Y1",
            "Y2"
        ]

        dfs.append(df)


    # 데이터 병합
    data = pd.concat(dfs, ignore_index=True)


    # 결측치 제거
    data = data.dropna()


    # 숫자형 변환
    numeric_cols = [
        "X1",
        "X2",
        "X3",
        "Y1",
        "Y2"
    ]


    data[numeric_cols] = data[numeric_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )


    # 변환 실패 제거
    data = data.dropna()



    # =========================
    # 기초 통계량
    # =========================

    describe_result = data.describe()



    # =========================
    # 상관계수
    # =========================

    corr = data[numeric_cols].corr()



    # =========================
    # VIF 계산
    # =========================

    X = data[
        [
            "X1",
            "X2",
            "X3"
        ]
    ]


    X_const = sm.add_constant(X)


    vif = pd.DataFrame()

    vif["Variable"] = X_const.columns


    vif["VIF"] = [
        variance_inflation_factor(
            X_const.values,
            i
        )
        for i in range(X_const.shape[1])
    ]



    # =========================
    # 회귀분석 함수
    # =========================

    def fit_model(target):

        y = data[target]


        model = sm.OLS(
            y,
            X_const
        ).fit()



        # 예측값

        pred = model.predict(
            X_const
        )



        # =====================
        # 실제값-예측값 그래프
        # =====================

        os.makedirs(
            "images",
            exist_ok=True
        )


        plt.figure(
            figsize=(6,6)
        )


        plt.scatter(
            y,
            pred
        )


        plt.xlabel(
            "Actual"
        )

        plt.ylabel(
            "Predicted"
        )


        plt.title(
            f"{target} Actual vs Predicted"
        )


        prediction_path = (
            f"images/{target}_prediction.png"
        )


        plt.savefig(
            prediction_path
        )

        plt.close()



        # =====================
        # 잔차 그래프
        # =====================

        residual = y - pred


        plt.figure(
            figsize=(6,5)
        )


        plt.scatter(
            pred,
            residual
        )


        plt.axhline(
            0,
            linestyle="--"
        )


        plt.xlabel(
            "Predicted"
        )


        plt.ylabel(
            "Residual"
        )


        plt.title(
            f"{target} Residual Plot"
        )


        residual_path = (
            f"images/{target}_residual.png"
        )


        plt.savefig(
            residual_path
        )

        plt.close()



        # 결과 반환

        return {

            "R_squared":
                float(model.rsquared),


            "Adj_R_squared":
                float(model.rsquared_adj),


            "p_values":
                model.pvalues.to_dict(),


            "prediction_image":
                prediction_path,


            "residual_image":
                residual_path
        }



    # =========================
    # Y1, Y2 분석
    # =========================

    result_Y1 = fit_model(
        "Y1"
    )


    result_Y2 = fit_model(
        "Y2"
    )



    # =========================
    # 최종 반환
    # =========================

    return {

        "describe":
            describe_result.to_dict(),


        "correlation":
            corr.to_dict(),


        "vif":
            vif.to_dict(),


        "Y1":
            result_Y1,


        "Y2":
            result_Y2
    }
