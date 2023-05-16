from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from helper import *
import pickle
import soundfile 
import librosa 
import numpy as np
import os
from deepface import DeepFace
from PIL import Image

app = Flask(__name__)
cors = CORS(app)
@app.route('/',methods=['GET'])
def voice_recog():
    model = pickle.load(open('model.pkl', 'rb'))
    index = request.args.get('index')
    take_input(index)
    sample = extract_feature('00-01-01-.wav', mfcc=True, chroma=True, mel=True)
    sample=sample.reshape(1,-1)
    y_pred=model.predict(sample.reshape(1,-1))
    return y_pred[0]


def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
            result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

@app.route("/face",methods=['POST'])
def face_rec():
    data=request.json
    image=data.get('image')

    result = DeepFace.find(img_path = image, db_path = "./datasets/",model_name='Facenet512',enforce_detection=False)[0]

    print(result)
    if result.empty:
        return {"status":"user_not_found"}
    else:
        return {"status":"user_found"}


if __name__ == '__main__':
    app.run(port=8080, debug=True,host="0.0.0.0")