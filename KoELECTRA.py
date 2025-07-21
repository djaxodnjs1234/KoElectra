# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 모델과 토크나이저 로드
model_name = "Jinuuuu/KoELECTRA_fine_tunning_emotion"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 감정 분석 함수
def analyze_emotion(text):
    # 토크나이징
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 확률 계산
    probs = torch.softmax(outputs.logits, dim=1)
    
    # 감정 레이블
    emotion_labels = ['angry', 'anxious', 'embarrassed', 'happy', 'heartache', 'sad']
    
    # 결과 반환
    emotion_probs = {}
    for i, label in enumerate(emotion_labels):
        emotion_probs[label] = float(probs[0][i])
    
    return emotion_probs

# 사용 예시
text = "오늘은 정말 행복한 하루였다."
result = analyze_emotion(text)

print("감정 분석 결과:")
for emotion, prob in sorted(result.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {prob:.3f}")

# %%
from transformers import pipeline

# 파이프라인 생성
classifier = pipeline(
    "text-classification",
    model="Jinuuuu/KoELECTRA_fine_tunning_emotion",
    tokenizer="Jinuuuu/KoELECTRA_fine_tunning_emotion"
)

# 감정 분석
texts = [
    "오늘은 정말 행복한 하루였다.",
    "너무 화가 나서 참을 수 없다.",
    "내일 시험이 걱정된다.",
    "너무 잠이 오지만 이 모델은 완성할 수 있다는 생각에 희망을 얻는다",
    "배가 고프다",
    "조직에서 배신당했다",
    "초코민트을 먹으니 우울해졌다"
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"텍스트: {text}")
    print(f"감정: {result['label']} (확률: {result['score']:.3f})")
    print()

# %%모델과 토크나이저 불러오기
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %%저장할 디렉토리 지정
save_directory = "./exported_model"

# %%모델과 토크나이저 저장
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)