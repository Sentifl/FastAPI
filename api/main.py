import os
from fastapi import FastAPI, Request, Depends, HTTPException, status
import requests
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import boto3
import dotenv
from io import BytesIO
import jwt

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

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("CREDENTIALS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("CREDENTIALS_SECRET_KEY"),
    region_name=os.getenv("S3_REGION")
)

@app.post("/create/music")
async def create_music(input:Request):
    input = await input.json()
    userId = input['user_id']
    htmlURL = input['html_url']
    token = input['token']
    
    verifyToken(token)
    
    #노래 생성 로직 추가 예정
    
    file_name = f"{uuid4()}.mp3"
    audio_url = saveMusicAtS3(file_name, userId)
    
    return {
            "emotion": "테스트용emotion",
            "url": audio_url,
            "title": file_name
            }
    
def saveMusicAtS3(file_name: str, userId: str) -> str:
    BUCKET_NAME = os.getenv("S3_BUCKET")
    
    try: 
        response = requests.get("https://sentifl-demo.s3.ap-northeast-2.amazonaws.com/tempUserId/music/48bea410-4d7b-467e-8603-dfd25de582ad.mp3")
        response.raise_for_status()
        
        audio_file = BytesIO(response.content)

        
        s3_key = f"music/{userId}/{file_name}"
        s3_client.upload_fileobj(
            audio_file, 
            BUCKET_NAME, 
            s3_key,
            ExtraArgs={'ContentType': 'audio/mp3'}
        )
        
        audio_url = f'https://{BUCKET_NAME}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_key}'
        
        return audio_url
    
    except boto3.exceptions.S3UploadFailedError as e:
        print(f"S3 업로드 실패: {e}")
        raise e
    
    except Exception as e:
        print(f"오류 발생: {e}")
        raise e
    
    
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