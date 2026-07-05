from utils.bert_predict import predict_proba
from utils.config import LABELS

samples = [
    'I got selected for my dream internship.',
    'I failed my exam and feel disappointed.',
    'Nobody listens to my ideas.',
    "I'm scared I might fail tomorrow's exam.",
    'Today I attended classes and completed my assignments.',
]

with open('bert_prediction_check.txt', 'w', encoding='utf-8') as fh:
    for s in samples:
        p = predict_proba(s)
        fh.write(s + '\n')
        fh.write('probs=' + str(p) + '\n')
        fh.write('pred=' + str(LABELS[int(p.argmax())]) + '\n')
        fh.write('---\n')
