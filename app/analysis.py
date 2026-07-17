import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from mpl_toolkits.mplot3d import Axes3D

import io
import base64



def run_analysis(file_list):

    dfs = []

    for file in file_list:

        df = pd.read_excel(file)

        df.columns = [
            "지역",
            "X1",
            "X2",
            "X3",
            "Y1",
            "Y2"
        ]

        dfs.append(df)



    data = pd.concat(
        dfs,
        ignore_index=True
    )


    data = data.dropna()


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


    data = data.dropna()



    # =========================
    # 기초 통계
    # =========================

    describe_result = data.describe()



    # =========================
    # 상관계수
    # =========================

    corr = data[numeric_cols].corr()



    # =========================
    # VIF
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
        for i in range(
            X_const.shape[1]
        )
    ]



    # =========================
    # 회귀 + K-means
    # =========================

    def fit_model(target):


        # =====================
        # 회귀 분석
        # =====================

        y = data[target]


        model = sm.OLS(
            y,
            X_const
        ).fit()


        pred = model.predict(
            X_const
        )



        # =====================
        # K-means 3차원
        # X1, X2, Y
        # =====================

        k_data = data[
            [
                "X1",
                "X2",
                target
            ]
        ]


        scaler = StandardScaler()


        scaled = scaler.fit_transform(
            k_data
        )



        # 최적 n 찾기

        best_k = 2
        best_score = -1


        for k in range(2, 11):

            temp = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )


            temp_cluster = temp.fit_predict(
                scaled
            )


            score = silhouette_score(
                scaled,
                temp_cluster
            )


            if score > best_score:
                best_score = score
                best_k = k



        # 최적 n으로 학습

        kmeans = KMeans(
            n_clusters=best_k,
            random_state=42,
            n_init=10
        )


        cluster = kmeans.fit_predict(
            scaled
        )



        # =====================
        # 그래프 생성
        # =====================

        fig = plt.figure(
            figsize=(8,12)
        )


        # 위 : 회귀

        ax0 = fig.add_subplot(
            211
        )


        ax0.scatter(
            y,
            pred
        )


        ax0.set_xlabel(
            "Actual"
        )


        ax0.set_ylabel(
            "Predicted"
        )


        ax0.set_title(
            f"{target} Regression"
        )



        # 아래 : 3D K-means

        ax1 = fig.add_subplot(
            212,
            projection="3d"
        )


        ax1.scatter(
            data["X1"],
            data["X2"],
            data[target],
            c=cluster,
            s=50
        )


        ax1.set_xlabel(
            "X1"
        )


        ax1.set_ylabel(
            "X2"
        )


        ax1.set_zlabel(
            target
        )


        ax1.set_title(
            f"K-means 3D ({target}), n={best_k}"
        )



        plt.tight_layout()



        # =====================
        # 그래프 base64 변환
        # =====================

        buffer = io.BytesIO()


        plt.savefig(
            buffer,
            format="png",
            bbox_inches="tight"
        )


        buffer.seek(0)


        graph_image = base64.b64encode(
            buffer.read()
        ).decode()



        plt.close()



        return {

            "R_squared":
                float(model.rsquared),


            "Adj_R_squared":
                float(model.rsquared_adj),


            "p_values":
                model.pvalues.to_dict(),


            "graph":
                graph_image,


            "clusters":
                cluster.tolist(),


            "optimal_k":
                best_k,


            "silhouette_score":
                float(best_score),


            "cluster_centers":
                kmeans.cluster_centers_.tolist()

        }



    # =========================
    # Y1 / Y2
    # =========================

    result_Y1 = fit_model(
        "Y1"
    )


    result_Y2 = fit_model(
        "Y2"
    )



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
