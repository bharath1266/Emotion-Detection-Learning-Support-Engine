Deployment Guide

Streamlit Cloud

1. Ensure `requirements.txt` lists all dependencies.
2. Push repository to GitHub.
3. On Streamlit Cloud, create a new app and point to this repo and the `app.py` file.
4. Set any required environment variables (see `.env.example`) in the Streamlit Cloud settings.

Local deployment

1. Create a virtual environment and install requirements (see README).
2. Place trained models under `models/bltsm/` and `models/bert_emotion_model_final/`.
3. Run `streamlit run app.py` and open the provided URL.

Notes

- For BERT training or inference on larger datasets, use a GPU-enabled environment (Colab, AWS/GCP instance, or local GPU). On native Windows, TensorFlow may not detect GPU without special drivers; consider WSL2 or cloud GPU.
- Keep large data and model artifacts out of version control — use `models/` and `data/` storage solutions or Git LFS when needed.
