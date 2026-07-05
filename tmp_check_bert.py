from utils.bert_predict import predict_proba
from utils.config import LABELS
samples = [
    'I got selected for my dream internship.',
    'I failed my exam and feel disappointed.',
    'Nobody listens to my ideas.',
    "I'm scared I might fail tomorrow's exam.",
    'Today I attended classes and completed my assignments.'
]
for s in samples:
    probs = predict_proba(s)
    print(s)
    print(probs)
    print(LABELS[int(probs.argmax())])
    print('---')
