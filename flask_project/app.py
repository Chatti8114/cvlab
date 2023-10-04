#### Run

# 1. conda activate hci_test
# 2. FLASK_APP=app
# 3. FLASK_ENV=development
# 4. flask run

#### Close

# 1. lsof -i :5000 (check pid)
# 2. kill -9 [pid number]

#### Import
from flask import Flask, render_template, request, jsonify

import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

import torch
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer

from transformers import Wav2Vec2FeatureExtractor
from speech_emotion import Wav2Vec2ForSpeechClassification  #myspeech

import nemo.collections.asr as nemo_asr
from transformers import AutoTokenizer, RobertaForSequenceClassification

import face_emotion as myface
import speech_emotion as myspeech
import image_retrieval as myretrieval

import os
import time


app = Flask(__name__, static_folder='./static/')

#### Load Model

device= 'cuda' if torch.cuda.is_available() else 'cpu'

##-----face_emotion-----##
detectionnet = MTCNN(keep_all=False,
                     post_process=False,
                     min_face_size=40,
                     device=device)
fernet = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf',
                             device=device)

##-----sound emotion-----##
sernet = Wav2Vec2ForSpeechClassification.from_pretrained("jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition3")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

##-----text emotion-----##
STTnet = nemo_asr.models.ASRModel.from_pretrained(model_name="eesungkim/stt_kr_conformer_transducer_large")
auto_tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
roberta_classifier = RobertaForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_dataset3_best")

print("\n==========Finished loading the model==========\n")


#### main

# save nearest node
global nearest_index
nearest_index = 10 # random_int_number

@app.route('/')
def index():
    # initial random netrual image
    nearest_index, random_nertral_img_path = myretrieval.initial_image_retrieval() 
    return render_template('index.html', image_retrieval=random_nertral_img_path)

@app.route('/upload', methods=['POST'])
def upload_video():
    video_file = request.files['video']

    if video_file:
        # measure running time
        start_time = time.time()

        video_path = os.path.join(os.getcwd(), 'user_data', 'video.webm')
        video_file.save(video_path)

        # convert video to MP4 and extract audio using ffmpeg
        video_output_path = os.path.join(os.getcwd(), 'user_data', "video.mp4")
        audio_output_path = os.path.join(os.getcwd(), 'user_data', "audio.wav")
        
        subprocess.run(['ffmpeg', '-y', '-i', video_path, '-q:v', '0', video_output_path, '-q:a', '0', audio_output_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)


        # get emotion
        user_emotion_dict = get_user_emotion(video_output_path, audio_output_path)
        
        # retrieve image
        user_emotion = user_emotion_dict["merged_emotion"]
        
        # save nearest node
        global nearest_index
        nearest_index, nearest_image_path = myretrieval.retrieve_image(user_emotion, nearest_index)
        print('* nearest_index : ', nearest_index)
        print("* nearest_image_path : ", nearest_image_path)
        
        # calculate running_time
        end_time = time.time()
        running_time = round(end_time - start_time, 2)
        print(f"* running time : {running_time} s")
        
        return jsonify(status="success", imagePath=nearest_image_path, emotions=user_emotion_dict, timeStamp=end_time)
    
    return jsonify(status="error")

def get_user_emotion(video_path, audio_path, detectionnet=detectionnet, fernet=fernet, sernet=sernet, STTnet=STTnet, auto_tokenizer=auto_tokenizer, roberta_classifier=roberta_classifier):
    
    # thread programming
    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        # 각 함수를 병렬로 실행
        future_face  = executor.submit(myface.get_face_emotion, video_path, detectionnet, fernet)
        future_sound = executor.submit(myspeech.get_sound_emotion, audio_path, feature_extractor, sernet)
        future_text  = executor.submit(myspeech.get_text_emotion, audio_path, STTnet, auto_tokenizer, roberta_classifier)
        
        # 각 함수의 결과값
        face_emotion_result  = future_face.result()
        sound_emotion_result = future_sound.result()
        text_from_speech, text_emotion_result  = future_text.result()
    
    
    print("* dialogue : ", text_from_speech)
    
    print("* face : ", face_emotion_result)
    print("* sound : ", sound_emotion_result)
    print("* text : ", text_emotion_result)
    
    user_emotion = myretrieval.merge_emotion_result(face_emotion_result, sound_emotion_result, text_emotion_result)
    print("* merged : ", user_emotion)
    
    user_emotion_dict = {
        "face_emotion": face_emotion_result,
        "sound_emotion": sound_emotion_result,
        "text_emotion": text_emotion_result,
        "merged_emotion": user_emotion
    }
    
    return user_emotion_dict

if __name__ == '__main__':
    app.run(debug=True)