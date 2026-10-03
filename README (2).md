# Modality-Attribution Faithfulness Demo

Live demo of the image+metadata fusion model (EfficientNet-B3 + MetaBlock
gating) from our research on modality-attribution faithfulness for skin
lesion classification (ISIC-DICM-17K).

Upload a skin lesion image + patient metadata to get:
- A live prediction (Fusion model and MetaBlock model)
- A live comparison of what each attribution method (Ablation ground truth,
  Integrated Gradients, MetaBlock gate weight) says is "important" for that
  prediction

This demo exists to show a working, deployed version of the trained model.
The validated, paper-reported results (across 3 datasets, 10-seed robustness
checks) are reported separately in the manuscript.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Files

- `streamlit_app.py` — app code
- `requirements.txt` — dependencies
- `ckpts/fusion_seed1.pth`, `ckpts/metablock_seed1.pth` — trained model weights (seed 1)

## Known limitation

The metadata branch expects 20 input dimensions; 4 of these (relating to
training-time categorical encoding) could not be reconstructed from
available files and are zero-padded. The image branch uses the exact
trained weights.
