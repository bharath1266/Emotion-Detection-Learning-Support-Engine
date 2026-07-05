import importlib

modules = [
    'utils.config',
    'utils.preprocessing',
    'utils.bilstm_predict',
    'utils.bert_predict',
    'utils.unified_predictor',
    'utils.gemini_helper',
    'utils.history_logger',
    'utils.csv_handler',
]

for name in modules:
    try:
        importlib.import_module(name)
        print(f'IMPORT_OK {name}')
    except Exception as exc:
        print(f'IMPORT_FAIL {name}: {exc}')
