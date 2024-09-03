import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ai import summary, emotion
from audiocraft.models import musicgen
import boto3
import dotenv
from io import BytesIO
from uuid import uuid4
from scipy.io.wavfile import write
import translate
import numpy as np
import torch

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
    
model = musicgen.MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=10)

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
    predictedEmotion = emotion.emotion_predict(blogContent)
    prompt = predictedEmotion + "의 감정이 나타나고 '" + summaryContent + "' 이 문장의 분위기를 잘 나타내는 음악. 장르는 상관 없다."
    translated_prompt = translate._translate(prompt)
    
    generated_music = model.generate([translated_prompt]) 
    
    file_name = f"{uuid4()}.mp3"

    if isinstance(generated_music, (list, tuple)):
        audio_url = saveMusicAtS3(generated_music[0], file_name, userId)
    else:
        audio_url = saveMusicAtS3(generated_music, file_name, userId)
    
    return {
            "emotion": predictedEmotion,
            "music_url": audio_url,
            "music_title": file_name
            }

def tensor_to_audio(tensor: torch.Tensor, sample_rate: int = 22050) -> np.ndarray:
    waveform = tensor.detach().cpu().numpy()
    if len(waveform.shape) == 2:
        waveform = np.mean(waveform, axis=0)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform, sample_rate

def saveMusicAtS3(generated_music: torch.Tensor, file_name: str, userId: str) -> str:
    BUCKET_NAME = os.getenv("S3_BUCKET")

    try:
        waveform, sample_rate = tensor_to_audio(generated_music)

        tmp_dir = "./tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            
        file_path = f"./tmp/{file_name}"
        write(file_path, sample_rate, waveform)

        s3_key = f"{userId}/music/{file_name}"
        s3_client.upload_file(
            file_path, 
            BUCKET_NAME, 
            s3_key,
            ExtraArgs={'ContentType': 'audio/mp3'}
        )

        os.remove(file_path)
        
        audio_url = f'https://{BUCKET_NAME}.s3.{s3_client.meta.region_name}.amazonaws.com/{s3_key}'
        return audio_url
    
    except boto3.exceptions.S3UploadFailedError as e:
        print(f"S3 업로드 실패: {e}")
        raise e
    except Exception as e:
        print(f"오류 발생: {e}")
        raise e