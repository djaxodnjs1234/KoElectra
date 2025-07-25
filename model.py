# %%
import joblib  # 또는 pickle, torch 등 사용한 방법에 따라
import os
import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast

#모델과 벡터라이저 로딩
try:
    model = joblib.load("text_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print(f"[ERROR] 모델 로딩 실패: {e}")

model.eval()  # 추론 모드로 설정

# 예측 함수
def predict(text: str) -> str:
    inputs = vectorizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)# 확률 계산
        probs = torch.softmax(outputs.logits, dim=1)
        
        # 감정 레이블
        emotion_labels = ['화남', '불안', '당황', '행복', '상심', '슬픔']
        # 결과 반환
        emotion_probs = {}
        for i, label in enumerate(emotion_labels):
            emotion_probs[label] = float(probs[0][i])
        # 확률에 100 곱하고 소수점 둘째 자리로 반올림
        emotion_probs_percent = {k: round(v * 100, 2) for k, v in emotion_probs.items()}
        top2 = sorted(emotion_probs_percent.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_dict = dict(top2)
    return str(top2_dict)



# %%
test_text = "오늘 너무 짜증나!"
result = predict(test_text)
print(f"입력: {test_text}")
print(f"예측된 감정: {result}")

# %%
