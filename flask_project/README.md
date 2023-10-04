# multimodal_interaction_flask
- We created an application that retrieves for image similar to the emotional values expressed.
- Emotional values are calculated by considering all three modal: face, speech, text(STT)
- Image retrieval uses a graph DB created through a generative model.
---
## Run Location
- multimodal_interaction/flask_project
- Where app.py is located.

## Set Virtual Enviroment
- pip install -r requirements.txt
- In the event of a crash error, Please set using direct_requirements.txt

## Run Application (in terminal)
- $ conda activate hci_test
- $ flask run
- Access to "http://127.0.0.1:5000/"
  
  ![Screenshot from 2023-08-24 17-57-57](https://github.com/kuai-lab/multimodal_interaction/assets/86465983/00a30ecc-c44f-4c10-9b5b-1b2542a213d1)

## Emotional Analasis and Image Retrieval
- Click "감정 녹화 및 분석 시작" and Start recording video for 5 seconds.
- Express your emotions during recording video.
- End of Recording ➔ Analyze Emotions ➔ Retrieve Image
- You can check the previous results by clicking target in "분석 결과 목록"
#### Result
  ![Screenshot from 2023-08-24 18-03-03](https://github.com/kuai-lab/multimodal_interaction/assets/86465983/a9c0461d-2a12-49e9-b37e-42e47e991164)
  - Live Video / Recoded Video / Retrieved Image / "분석 결과 목록"
  - Face Emotion / Sound Emotion / Text Emotion / Merged Emotion

  ![Screenshot from 2023-08-24 18-01-50](https://github.com/kuai-lab/multimodal_interaction/assets/86465983/1f31fb4d-5119-4d59-99b6-a0a0997b8b3d)
