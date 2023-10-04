#### Import

import cv2
# import torch
# from facenet_pytorch import MTCNN
# from hsemotion.facial_emotions import HSEmotionRecognizer


#### Functions

# get bounding_boxes (using frame)
def detect_face(frame, detectionnet):
    bounding_boxes, probs = detectionnet.detect(frame, landmarks=False)
    try:
        bounding_boxes = bounding_boxes[probs > 0.8]  # initial thres = 0.9
    except:
        bounding_boxes = []
    return bounding_boxes

# get face_emotion (using boundingboxes)
def predict_face_emotion(frame, detectionnet, fernet):
    face_emotion_result = {}    
    class_label = {0: 'anger',
                   1: 'contempt',
                   2: 'disgust',
                   3: 'fear',
                   4: 'happy',
                   5: 'neutral',
                   6: 'sadness',
                   7: 'surprise'}
    face_emotion_result = dict.fromkeys(class_label.values(), 0)  # 모든 감정 값을 0으로 설정

    try:
        ## detect face (using function)
        bounding_boxes = detect_face(frame, detectionnet)
        for bbox in bounding_boxes:
            
            ### crop face
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[0:4]
            face_img = frame[y1:y2, x1:x2, :]
            
            ### predict face emotion
            emotions, scores = fernet.predict_emotions(face_img, logits=False)
            face_emotion_result = dict(zip(class_label.values(), list(round(float(i), 4) for i in scores)))

    except Exception as e:
        print("Facial Emotion Analysis Failed")

    return face_emotion_result

def calculate_average(emotion_list):
    total_counts = {}
    summed_values = {}
    
    for data in emotion_list:
        for key, value in data.items():
            summed_values[key] = summed_values.get(key, 0) + value
            total_counts[key] = total_counts.get(key, 0) + 1
    
    averages = {}
    for key in summed_values:
        averages[key] = summed_values[key] / total_counts[key]
    
    return averages

def calculate_max_ratio(emotion_list):
    # 모든 가능한 키를 추출
    all_keys = list(emotion_list[0].keys())

    # 각 딕셔너리에서 최대값을 가진 키를 추출
    max_keys = [max(d, key=d.get) for d in emotion_list if d]

    # 각 키별로 몇 번 등장하는지 세기
    key_counts = {key: 0 for key in all_keys}
    for key in max_keys:
        key_counts[key] += 1

    # 비율 계산
    total = len(max_keys)
    max_ratio = {key: count / total for key in all_keys for count in [key_counts[key]]}

    return max_ratio


def get_face_emotion(video_path, detectionnet, fernet, frame_unit=5): # 5frame/1s
    # frame unit -> interval
    interval = 1 / frame_unit

    # load video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    frame_interval = int(fps * interval)
    frame_count = 0

    # get face emotion by frame
    face_emotion_list = []  # 감정을 저장할 리스트

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # predict emotion per frame
        if frame_count % frame_interval == 0:
            face_emotion_result = predict_face_emotion(frame, detectionnet, fernet)
            face_emotion_list.append(face_emotion_result)
        frame_count += 1

    # close video
    cap.release()
    
    # calculate frame integration
    text_emotion_result = calculate_average(face_emotion_list)
    
    return text_emotion_result


# def get_face_emotion_by_frame(video_path, detectionnet, fernet, frame_unit=5):
#     # frame unit -> interval
#     interval = 1 / frame_unit

#     # load video
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     frame_interval = int(fps * interval)
#     frame_count = 0

#     # get face emotion by frame
#     face_emotion_list = []  # 감정을 저장할 리스트

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # predict emotion per frame
#         if frame_count % frame_interval == 0:
#             face_emotion_result = predict_face_emotion(frame, detectionnet, fernet)
#             face_emotion_list.append(face_emotion_result)
#         frame_count += 1

#     # close video
#     cap.release()
    
#     return face_emotion_list