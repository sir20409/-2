from app.analysis import run_analysis


# 분석 결과 저장용
analysis_result = {}


def initialize_analysis():
    """
    서버 시작 시 자동 분석 실행
    """

    global analysis_result


    files = [
        "data/21년도 데이터.xlsx",
        "data/22년도 데이터.xlsx",
        "data/23년도 데이터.xlsx",
        "data/24년도 데이터.xlsx"
    ]


    analysis_result = run_analysis(files)



def get_analysis_result():
    """
    분석 결과 반환
    """

    return analysis_result
