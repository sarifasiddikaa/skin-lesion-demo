# MelaXAI: Skin Lesion Research Prototype

A public, browser-based research prototype for the paper
**"Predictive performance and modality attribution in melanoma classification using dermoscopic images and clinical metadata"** (manuscript in preparation).

> **Research prototype only. Not for clinical use or diagnosis. Not clinically validated.**

**Live demo:** https://skin-lesion-demo-sst.streamlit.app/

## What it does

You upload a dermoscopic image and enter some patient details. The app shows:

- the predicted melanoma probability from two models, **Fusion** and **MetaBlock**
- a comparison of how much the image and the metadata contributed to the result, using three methods: an ablation-derived reference, Integrated Gradients, and the MetaBlock gate

## Inputs

- Dermoscopic image (png, jpg, jpeg)
- Sex
- Anatomical site
- Personal history of melanoma
- Family history of melanoma
- Age
- Lesion size (mm)

## Models

- Image encoder: EfficientNet-B3
- **Fusion:** image features and metadata features are concatenated
- **MetaBlock:** metadata gates and shifts the image features
- Trained on the ISIC-DICM-17K dataset
- Checkpoints are in the `ckpts/` folder

## Run it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py`: the app
- `ckpts/`: trained model checkpoints
- `requirements.txt`: Python packages

## Disclaimer

This tool is for research and demonstration only. It is not a medical device and must not be used for diagnosis or patient care. Always consult a dermatologist for clinical evaluation.
