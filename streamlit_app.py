"""
Live demo: Modality-Attribution Faithfulness for Skin Lesion Fusion Models.
Streamlit version for permanent free hosting on Streamlit Community Cloud.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image
import streamlit as st
from captum.attr import IntegratedGradients

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = os.path.join(os.path.dirname(__file__), 'ckpts')

SEX_OPTIONS = ['male', 'female']
SITE_OPTIONS = [
    'anterior torso', 'head/neck', 'lateral torso', 'lower extremity',
    'oral/genital', 'palms/soles', 'posterior torso', 'upper extremity',
]
HX_OPTIONS = ['No', 'Yes']

REF_COLS_14 = (
    [f'sex_{s}' for s in SEX_OPTIONS] +
    [f'anatom_site_general_{s}' for s in SITE_OPTIONS] +
    ['personal_hx_mm_0.0', 'personal_hx_mm_1.0'] +
    ['family_hx_mm_0.0', 'family_hx_mm_1.0']
)
META_DIM = 20
AGE_MEDIAN, SIZE_MEDIAN = 60.0, 6.0


class FusionModel(nn.Module):
    def __init__(self, meta_dim, num_classes=2):
        super().__init__()
        eff = tv_models.efficientnet_b3(weights=None)
        self.image_branch = nn.Sequential(eff.features, eff.avgpool)
        self.meta_branch = nn.Sequential(nn.Linear(meta_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(1536 + 32, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, img, meta):
        i = self.image_branch(img).flatten(1)
        m = self.meta_branch(meta)
        return self.classifier(torch.cat([i, m], dim=1))


class MetaBlockModel(nn.Module):
    def __init__(self, meta_dim, num_classes=2):
        super().__init__()
        eff = tv_models.efficientnet_b3(weights=None)
        self.image_branch = nn.Sequential(eff.features, eff.avgpool)
        self.meta_gate = nn.Sequential(nn.Linear(meta_dim, 1536))
        self.meta_bias = nn.Sequential(nn.Linear(meta_dim, 1536))
        self.classifier = nn.Sequential(nn.Linear(1536, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes))

    def forward(self, img, meta):
        i = self.image_branch(img).flatten(1)
        gate = torch.sigmoid(self.meta_gate(meta))
        bias = torch.tanh(self.meta_bias(meta))
        modulated = i * gate + bias
        return self.classifier(modulated), gate


@st.cache_resource
def load_models():
    fusion_model = FusionModel(META_DIM).to(DEVICE)
    fusion_sd = torch.load(os.path.join(CKPT_DIR, 'fusion_seed1.pth'), map_location=DEVICE, weights_only=False)
    fusion_model.load_state_dict(fusion_sd, strict=False)
    fusion_model.eval()

    mb_model = MetaBlockModel(META_DIM).to(DEVICE)
    mb_sd = torch.load(os.path.join(CKPT_DIR, 'metablock_seed1.pth'), map_location=DEVICE, weights_only=False)
    mb_model.load_state_dict(mb_sd, strict=False)
    mb_model.eval()
    return fusion_model, mb_model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def encode_metadata(sex, site, personal_hx, family_hx, age, size_mm):
    vec14 = np.zeros(len(REF_COLS_14), dtype='float32')
    vec14[REF_COLS_14.index(f'sex_{sex}')] = 1.0
    vec14[REF_COLS_14.index(f'anatom_site_general_{site}')] = 1.0
    vec14[REF_COLS_14.index(f'personal_hx_mm_{"1.0" if personal_hx == "Yes" else "0.0"}')] = 1.0
    vec14[REF_COLS_14.index(f'family_hx_mm_{"1.0" if family_hx == "Yes" else "0.0"}')] = 1.0
    numeric = np.array([float(age), float(size_mm)], dtype='float32')
    full_known = np.concatenate([numeric, vec14])
    pad_len = META_DIM - len(full_known)
    full20 = np.concatenate([full_known, np.zeros(pad_len, dtype='float32')])
    return torch.tensor(full20).unsqueeze(0).to(DEVICE)


def predict(fusion_model, mb_model, image, sex, site, personal_hx, family_hx, age, size_mm):
    img_t = transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    meta_t = encode_metadata(sex, site, personal_hx, family_hx, age, size_mm)

    with torch.no_grad():
        fusion_out = fusion_model(img_t, meta_t)
        fusion_prob = F.softmax(fusion_out, dim=1)[0, 1].item()

        mb_out, gate = mb_model(img_t, meta_t)
        mb_prob = F.softmax(mb_out, dim=1)[0, 1].item()
        gate_val = gate.mean().item()

        zero_img = torch.zeros_like(img_t)
        zero_meta = torch.zeros_like(meta_t)
        p_full = F.softmax(fusion_model(img_t, meta_t), dim=1)[0, 1].item()
        p_imgonly = F.softmax(fusion_model(img_t, zero_meta), dim=1)[0, 1].item()
        p_metaonly = F.softmax(fusion_model(zero_img, meta_t), dim=1)[0, 1].item()

    ablation_img_importance = p_full - p_metaonly
    ablation_meta_importance = p_full - p_imgonly

    def forward_fn(img, meta):
        return F.softmax(fusion_model(img, meta), dim=1)
    ig = IntegratedGradients(forward_fn)
    baseline_img = torch.zeros_like(img_t)
    baseline_meta = torch.zeros_like(meta_t)
    target_class = 1 if fusion_prob >= 0.5 else 0
    attr_img, attr_meta = ig.attribute((img_t, meta_t), baselines=(baseline_img, baseline_meta),
                                        target=target_class, n_steps=20)
    ig_img_mag = attr_img.abs().sum().item()
    ig_meta_mag = attr_meta.abs().sum().item()
    tot = ig_img_mag + ig_meta_mag + 1e-9
    ig_img_frac = ig_img_mag / tot

    return {
        'fusion_prob': fusion_prob, 'mb_prob': mb_prob, 'gate_val': gate_val,
        'ablation_img': ablation_img_importance, 'ablation_meta': ablation_meta_importance,
        'ig_img_frac': ig_img_frac,
    }


# ----------------------------- UI -----------------------------

st.set_page_config(
    page_title="Modality Attribution Faithfulness Demo",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .main > div { padding-top: 1.5rem; }
    .hero {
        background: linear-gradient(135deg, #5b3dd6 0%, #8b5cf6 100%);
        padding: 2rem 2.2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .hero h1 { margin: 0 0 0.4rem 0; font-size: 1.9rem; }
    .hero p { margin: 0; opacity: 0.92; font-size: 0.98rem; line-height: 1.5; }
    .card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        margin-bottom: 1rem;
    }
    .verdict-malignant {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ef4444;
    }
    .verdict-benign {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 1.15rem;
        font-weight: 700;
        color: #16a34a;
    }
    .disclosure-box {
        background: rgba(234, 179, 8, 0.1);
        border-left: 4px solid #eab308;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🔬 Modality-Attribution Faithfulness Demo</h1>
    <p>Live demo of the image + metadata fusion model (EfficientNet-B3 + MetaBlock gating)
    from our research on skin lesion classification (ISIC-DICM-17K). Upload a lesion image
    and patient metadata to get a live prediction, plus a live comparison of what each
    attribution method says is "important" versus the real, ablation-measured importance —
    the core question of our paper.</p>
</div>
""", unsafe_allow_html=True)

with st.expander("⚠️ Disclosure — read before interpreting results", expanded=False):
    st.markdown("""
    <div class="disclosure-box">
    The metadata branch's exact training-time encoding for 4 of its 20 input dimensions
    could not be reconstructed from available files; those 4 slots are zero-padded here.
    The <b>image branch is exact</b> (same weights as training). This demo exists to prove
    real deployment capability, not to reproduce the paper's exact reported numbers —
    see the paper for validated results across 3 datasets and 10-seed robustness checks.
    </div>
    """, unsafe_allow_html=True)

with st.spinner("Loading models..."):
    fusion_model, mb_model = load_models()

col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📋 Patient & lesion details")
    uploaded_file = st.file_uploader("Skin lesion image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), caption="Uploaded lesion", width=260)

    c1, c2 = st.columns(2)
    with c1:
        sex = st.selectbox("Sex", SEX_OPTIONS, index=1)
        personal_hx = st.radio("Personal history of melanoma", HX_OPTIONS, horizontal=True)
        age = st.slider("Age (approx)", 0, 90, 60)
    with c2:
        site = st.selectbox("Anatomical site", SITE_OPTIONS, index=6)
        family_hx = st.radio("Family history of melanoma", HX_OPTIONS, horizontal=True)
        size_mm = st.slider("Lesion size (mm)", 1, 100, 6)

    run_btn = st.button("▶ Run prediction + attribution", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if run_btn:
        if uploaded_file is None:
            st.error("Please upload a skin lesion image first.")
        else:
            image = Image.open(uploaded_file)
            with st.spinner("Running model..."):
                r = predict(fusion_model, mb_model, image, sex, site, personal_hx, family_hx, age, size_mm)

            is_malignant = r['fusion_prob'] >= 0.5
            verdict = "MALIGNANT (melanoma)" if is_malignant else "BENIGN"
            verdict_class = "verdict-malignant" if is_malignant else "verdict-benign"
            icon = "🔴" if is_malignant else "🟢"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="{verdict_class}">{icon} Prediction: {verdict}</div>', unsafe_allow_html=True)
            st.write("")

            m1, m2, m3 = st.columns(3)
            m1.metric("Fusion model", f"{r['fusion_prob']:.1%}", help="P(malignant)")
            m2.metric("MetaBlock model", f"{r['mb_prob']:.1%}", help="P(malignant)")
            m3.metric("Gate activation", f"{r['gate_val']:.3f}", help="Mean MetaBlock gate value")
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧭 Which input mattered more? (this one sample)")
            st.table({
                "Method": ["Ablation (ground truth)", "Integrated Gradients", "MetaBlock gate"],
                "Image importance": [f"{r['ablation_img']:+.3f}", f"{r['ig_img_frac']:.3f} (fraction)", "–"],
                "Metadata importance": [f"{r['ablation_meta']:+.3f}", f"{1-r['ig_img_frac']:.3f} (fraction)", f"{r['gate_val']:.4f}"],
            })
            st.caption(
                "Ablation shows the real, measured importance for this sample. Compare: does IG's "
                "fraction point the same direction? Does the gate value vary meaningfully for this "
                "patient, or does it look collapsed (narrow ~0.48–0.53 range) like our paper found?"
            )
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Fill in the details on the left and click **Run prediction + attribution** to see results here.")
        st.markdown('</div>', unsafe_allow_html=True)
