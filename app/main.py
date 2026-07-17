from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.router import router
from app.service import initialize_analysis


app = FastAPI(
    title="범죄 데이터 분석 시스템"
)


app.mount(
    "/images",
    StaticFiles(directory="images"),
    name="images"
)


app.include_router(router)



@app.on_event("startup")
def startup_event():
    initialize_analysis()



@app.get("/", response_class=HTMLResponse)
def home():

    return """
    <html>
    <body>

    <h1>범죄 데이터 분석 결과</h1>

    <h2>Y1 보이스피싱</h2>
    <img src="/images/Y1_prediction.png" width="500">
    <img src="/images/Y1_residual.png" width="500">


    <h2>Y2 인터넷 사기</h2>
    <img src="/images/Y2_prediction.png" width="500">
    <img src="/images/Y2_residual.png" width="500">

    </body>
    </html>
    """
