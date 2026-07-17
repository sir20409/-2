from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from app.router import router
from app.service import initialize_analysis, get_analysis_result


app = FastAPI(
    title="범죄 데이터 분석 시스템"
)


app.include_router(router)



@app.on_event("startup")
def startup_event():
    initialize_analysis()



@app.get("/", response_class=HTMLResponse)
def home():

    result = get_analysis_result()

    return f"""
    <html>
    <body>

    <h1>범죄 데이터 분석 결과</h1>


    <h2>Y1 보이스피싱</h2>
    <img src="data:image/png;base64,{result['Y1']['graph']}" width="700">


    <h2>Y2 인터넷 사기</h2>
    <img src="data:image/png;base64,{result['Y2']['graph']}" width="700">


    </body>
    </html>
    """
