#### Import

import numpy as np
import pydub as pyau

import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.file_utils import ModelOutput

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)


#### Class (Wav2Vec2ForSpeechClassification)
@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

#### Functions
def get_sound_emotion(audio_file, feature_extractor, sernet):
    try:
        # preprocess audio file
        audio_segment = pyau.AudioSegment.from_wav(audio_file)
        audio_segment = audio_segment.set_frame_rate(16000)
        audio_samples = np.array(audio_segment.get_array_of_samples())

        # extract audio features
        extracted_features = feature_extractor( raw_speech=audio_samples,
                                            sampling_rate=16000,
                                            padding=True,
                                            return_tensors="pt")
        model_output = sernet.forward(extracted_features.input_values.float())

        # get sound emotion
        sound_emotion_id2label = {
                "0": "happy",
                "1": "embarrassment",
                "2": "anger",
                "3": "fear",
                "4": "sadness",
                "5": "neutral"
        }
        
        sound_emotion_result = dict(zip(sound_emotion_id2label.values(), 
                                        [round(float(value), 4) for value in F.softmax(model_output[0][0], dim=0)]))

    except Exception as e:
        print("Sound Emotion Analysis Failed")
        sound_emotion_result = dict.fromkeys(sound_emotion_id2label.values(), 0)  # 모든 감정 값을 0으로 설정

    return sound_emotion_result


def get_text_emotion(audio_file, STTnet, tokenizer, classifier):
    
    # Speech To Text
    try:
        transcription = STTnet.transcribe([audio_file])
        text_from_speech = transcription[0][0]
    except Exception as e:
        print("STT Failed")
        text_from_speech = ''

    # Text Analysis
    text_emotion_id2label = {
        "0": "neutral",
        "1": "happy",
        "2": "embarrassment",
        "3": "anger",
        "4": "fear",
        "5": "sadness",
        "6": "disgust",
    }
    
    try:
        input_ids = torch.tensor([tokenizer.encode(text_from_speech)])
        result = classifier(input_ids)
        text_emotion_result = dict(zip(text_emotion_id2label.values(), list(round(float(i), 4) for i in F.softmax(result[0][0], dim=0))))

    except Exception as e:
        print("Text Emotion Analysis Failed")
        text_emotion_result = dict.fromkeys(text_emotion_id2label.values(), 0)  # 모든 감정 값을 0으로 설정

    return text_from_speech, text_emotion_result