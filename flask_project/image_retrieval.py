#### Import

import numpy as np
import pandas as pd
import os

import torch
from sklearn.metrics import mean_absolute_error as mae

#### Functions
def merge_emotion_result(face_emotion_result, sound_emotion_result, text_emotion_result):    
    
    merged_emotion_result = {}
    
    # 감정별로 평균 계산 함수
    def average_emotion(*emotions):
        # 0이 아닌 값만 필터링
        valid_emotions = [e for e in emotions if e != 0]
        return sum(valid_emotions) / len(valid_emotions) if valid_emotions else 0

    merged_emotion_result['anger'] = average_emotion(face_emotion_result['anger'],
                                                     sound_emotion_result['anger'],
                                                     text_emotion_result['anger'])
    
    merged_emotion_result['disgust'] = average_emotion(face_emotion_result['disgust']+
                                                       face_emotion_result['contempt'],
                                                       text_emotion_result['disgust'])
    
    merged_emotion_result['fear'] = average_emotion(face_emotion_result['fear'],
                                                    sound_emotion_result['fear'],
                                                    text_emotion_result['fear'])

    merged_emotion_result['joy'] = average_emotion(face_emotion_result['happy'],
                                                   sound_emotion_result['happy'],
                                                   text_emotion_result['happy'])

    merged_emotion_result['neutral'] = average_emotion(face_emotion_result['neutral'],
                                                       sound_emotion_result['neutral'],
                                                       text_emotion_result['neutral'])
    
    merged_emotion_result['sadness'] = average_emotion(face_emotion_result['sadness'],
                                                       sound_emotion_result['sadness'],
                                                       text_emotion_result['sadness'])
    
    merged_emotion_result['surprise'] = average_emotion(face_emotion_result['surprise'],
                                                        sound_emotion_result['embarrassment'],
                                                        text_emotion_result['embarrassment'])

    return merged_emotion_result


## default path
database_csv = pd.read_csv("static/graph_data/clip_cluster0_toy_database.csv")
graph_dataset = torch.load('static/graph_data/clip_cluster0_toy_graph_data.pt')
image_folder_path = 'static/graph_data/diffusion_image_dataset/image'

def retrieve_image(user_emotion_dict, new_node_index, database_csv=database_csv, graph_dataset=graph_dataset, image_folder_path=image_folder_path):
    # Find neighbor_node position matching new_node_index
    neighbor_node_positons = (graph_dataset.edge_index[1] == new_node_index).nonzero(as_tuple=False).squeeze()
    # Find neighbor_node_index
    neighbor_node_indices = graph_dataset.edge_index[0, neighbor_node_positons].cpu().numpy()
    # Excluding myself
    neighbor_node_indices = neighbor_node_indices[neighbor_node_indices != new_node_index]
    print('* neighbor_node_indices : ', neighbor_node_indices)
    # Extract emotions of neighbor_node
    neighbor_node_emotions = []
    for neighbor_node_i in neighbor_node_indices:
        neighbor_node_emotions.append(np.array(database_csv.iloc[neighbor_node_i]['anger':'surprise'].to_list()))
    # Transform to list
    user_emotion = np.array(list(user_emotion_dict.values()))
    # Calculate the distance in emotion value
    distances = []
    for neighbor_node_emotion in neighbor_node_emotions:
                distances.append(mae(neighbor_node_emotion, user_emotion))
    # Select nearest_node_index
    nearest_index = neighbor_node_indices[distances.index(min(distances))]
    # Image path of the nearest_node
    nearest_image_name = database_csv.iloc[nearest_index]['img_path']
    # Set up an image path
    nearest_image_path = os.path.join(image_folder_path, nearest_image_name)
    
    return nearest_index, nearest_image_path

def initial_image_retrieval(database_csv=database_csv, image_folder_path=image_folder_path):
    neutral_rows = database_csv[database_csv['label'] == 'neutral']
    random_neutral_index = neutral_rows.sample(n=1)
    random_nertral_img_name = random_neutral_index['img_path'].iloc[0]
    random_nertral_img_path = os.path.join(image_folder_path, random_nertral_img_name)
    # static path error
    random_nertral_img_path = random_nertral_img_path[7:]
    
    return random_neutral_index, random_nertral_img_path
