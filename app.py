from flask import Flask, render_template
import pickle
import soundfile 
import librosa 
import numpy as np
import pyaudio
import wave

app = Flask(__name__)
WAVE_OUTPUT_FILENAME = "00-01-01-.wav"
@app.route('/',methods=['GET'])
def voice_recog():
    model = pickle.load(open('model.pkl', 'rb'))
    take_input()
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

def take_input():

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 5
    index = 1
    audio = pyaudio.PyAudio()
    print("recording via index "+str(index))

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    print ("recording started")
    Recordframes = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print ("recording stopped")  
    save_audio(Recordframes, FORMAT, CHANNELS, audio, RATE)

def save_audio(Recordframes, FORMAT, CHANNELS, audio, RATE):
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()   

if __name__ == '__main__':
    app.run(port=8080, debug=True)