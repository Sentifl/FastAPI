import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ai import summary, emotion
from audiocraft.models import musicgen
import boto3
import dotenv
from io import BytesIO
from uuid import uuid4
import scipy.io.wavfile as wavfile

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
    
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-medium')
model.set_generation_params(duration=30)

# S3 연결 준비
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("CREDENTIALS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("CREDENTIALS_SECRET_KEY"),
    region_name=os.getenv("S3_REGION")
)

@app.post("/create/music")
async def create_music(input:Request):
    input = await input.json()
    blogContent = input['content']
    userId = input['userId']
    
    summaryContent = summary.summary_text(blogContent)
    #predictedEmotion = emotion.emotion_predict(blogContent)
    #prompt = [predictedEmotion + "감정의 느낌을 주고 '" + summaryContent + "' 이 문장의 분위기를 잘 나타내는 음악. 장르는 상관 없다."]
    prompt = ["기쁜 감정의 느낌을 주고 '" + summaryContent + "' 이 문장의 분위기를 잘 나타내는 음악. 장르는 상관 없다."]
    
    generated_music = model.generate(prompt) 
    audio_array = generated_music[0].cpu().numpy()
    
    file_name = f"{uuid4()}.wav" 
    audio_url = saveMusicAtS3(audio_array, file_name, userId)
    
    return {#"emotion": predictedEmotion,
            "emotion": "임시감정",
            "music_url": audio_url,
            "music_title": file_name}

def saveMusicAtS3(audio_array, file_name, userId):
    BUCKET_NAME = os.getenv("S3_BUCKET")
    
    try:
        audio_buffer = BytesIO()
        wavfile.write(audio_buffer, 32000, audio_array)
        audio_buffer.seek(0)  # 버퍼의 시작으로 되돌리기
        
        s3_client.upload_fileobj(audio_buffer, BUCKET_NAME, f'{userId}/{file_name}', ExtraArgs={'ContentType': 'audio/wav'})
        s3_url = f'https://{BUCKET_NAME}.s3.{s3_client.meta.region_name}.amazonaws.com/{userId}/{file_name}'
       
        return s3_url
    
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        return None