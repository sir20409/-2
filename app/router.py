from fastapi import APIRouter
from app.service import get_analysis_result

router = APIRouter(
    tags=["분석 결과"]
)

@router.get(
    "/api/analysis",
    summary="범죄 데이터 회귀분석 결과 조회",
    description="Y1(보이스피싱), Y2(인터넷 사기)에 대한 회귀분석 결과 반환"
)
def analysis():
    """
    분석 결과 반환
    """
    return get_analysis_result()
