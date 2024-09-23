import os
from fastapi import FastAPI, Request, HTTPException, status
import requests
from fastapi.middleware.cors import CORSMiddleware
from ai import summary, emotion
from uuid import uuid4
import dotenv
import bs
import jwt
import json

dotenv.load_dotenv()

app = FastAPI()

origins = [ "*" ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/create/music")
async def create_music(input:Request):
    input = await input.json()
    userId = input['user_id']
    postURL = input['post_url']
    token = input['token']
    
    #토큰 검증
    verifyToken(token)
    
    #url로부터 html 코드 가져오기
    html = getHtml(postURL)
    
    #html 코드에서 텍스트 값만 가져오기
    blogContent = bs.parsePost(html)
    
    print("blogContent: "+blogContent)
    
    #요약, 감정분석, 노래 생성
    summaryContent = summary.summary_text(blogContent)
    print("summaryContent: "+summaryContent)
    emotions = emotion.emotion_predict(blogContent)
    top1Emotion, top2Emotion = emotions["emotions"]
    prompt = top1Emotion + ", " + top2Emotion + "의 감정이 나타나고 '" + summaryContent + "' 이 문장의 분위기를 잘 나타내는 음악. 장르는 상관 없다."
    translatedPrompt = bs._translate(prompt)
    fileName = f"{uuid4()}.mp3"
    
    colab_url = "https://delicate-toucan-sensible.ngrok.io/sentifl/musicGen"
    data = {
        "userId": userId,
        "prompt": translatedPrompt,
        "fileName": fileName
    }
    
    res = requests.post(
        url=colab_url,
        json=data
    )
    
    #응답 상태코드가 200OK가 아니라면, 예외 발생
    res.raise_for_status()
    
    response_data = res.json()
    musicUrl = response_data.get("musicUrl")
    
    return {
            "emotion1": top1Emotion,
            "emotion2": top2Emotion,
            "url": musicUrl,
            "title": fileName,
            }
    
def verifyToken(token: str):
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
    
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid token"
        )
        
def getHtml(postUrl: str) -> str:
    try:
        response = requests.get(postUrl)
        response.raise_for_status() 

        data = response.json()

        html_content = data.get("content", "")
        
        return html_content
    
    except Exception as e:
        return str(e)