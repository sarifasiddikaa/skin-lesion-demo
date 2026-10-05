"""
Skin lesion research prototype — image + metadata fusion models (Fusion and MetaBlock).
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
    res_f = fusion_model.load_state_dict(fusion_sd, strict=False)
    print('FUSION load:', res_f)  # Manage app > logs: both key lists must be empty
    fusion_model.eval()

    mb_model = MetaBlockModel(META_DIM).to(DEVICE)
    mb_sd = torch.load(os.path.join(CKPT_DIR, 'metablock_seed1.pth'), map_location=DEVICE, weights_only=False)
    res_m = mb_model.load_state_dict(mb_sd, strict=False)
    print('METABLOCK load:', res_m)  # Manage app > logs: both key lists must be empty
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
                                        target=target_class, n_steps=8, internal_batch_size=2)
    ig_img_mag = attr_img.abs().sum().item()
    ig_meta_mag = attr_meta.abs().sum().item()
    tot = ig_img_mag + ig_meta_mag + 1e-9
    ig_img_frac = ig_img_mag / tot

    return {
        'fusion_prob': fusion_prob, 'mb_prob': mb_prob, 'gate_val': gate_val,
        'ablation_img': ablation_img_importance, 'ablation_meta': ablation_meta_importance,
        'ig_img_frac': ig_img_frac,
    }


# ============================= UI =============================

st.set_page_config(
    page_title="MelaXAI — Skin Lesion Research Prototype",
    page_icon="🩺",
    layout="wide",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main > div { padding-top: 1.2rem; max-width: 1200px; }

    .topbar {
        display: flex; align-items: center; gap: 0.6rem;
        padding-bottom: 0.3rem; margin-bottom: 1.4rem;
        border-bottom: 1px solid rgba(128,128,128,0.2);
    }
    .topbar .logo {
        font-size: 1.55rem; font-weight: 800; color: #0f766e;
        letter-spacing: -0.02em;
    }
    .topbar .tag {
        font-size: 0.8rem; color: #64748b; padding-top: 0.3rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
    }

    .risk-pill {
        display: inline-flex; align-items: center; gap: 0.5rem;
        padding: 0.55rem 1.1rem; border-radius: 999px;
        font-size: 1.05rem; font-weight: 700;
    }
    .risk-high { background: rgba(220, 38, 38, 0.1); color: #dc2626; border: 1px solid rgba(220,38,38,0.3); }
    .risk-low  { background: rgba(5, 150, 105, 0.1); color: #059669; border: 1px solid rgba(5,150,105,0.3); }

    .section-label {
        text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.76rem;
        font-weight: 700; color: #64748b; margin-bottom: 0.4rem;
    }
    .footnote {
        font-size: 0.78rem; color: #94a3b8; line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="topbar">
    <span class="logo">🩺 MelaXAI </span>
    <span class="tag">Skin lesion research prototype</span>
</div>
""", unsafe_allow_html=True)

with st.spinner("Loading model..."):
    fusion_model, mb_model = load_models()

col1, col2 = st.columns([1, 1.25], gap="large")

with col1:
    with st.container(border=True):
        st.markdown('<div class="section-label">Patient & lesion details</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Lesion image", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        if uploaded_file is not None:
            st.image(Image.open(uploaded_file), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            sex = st.selectbox("Sex", SEX_OPTIONS, index=1)
            personal_hx = st.radio("Personal history of melanoma", HX_OPTIONS, horizontal=True)
            age = st.slider("Age", 0, 90, 60)
        with c2:
            site = st.selectbox("Anatomical site", SITE_OPTIONS, index=6)
            family_hx = st.radio("Family history of melanoma", HX_OPTIONS, horizontal=True)
            size_mm = st.slider("Lesion size (mm)", 1, 100, 6)

        run_btn = st.button("Analyze", type="primary", use_container_width=True)

with col2:
    if run_btn:
        if uploaded_file is None:
            st.error("Please upload a lesion image first.")
        else:
            image = Image.open(uploaded_file)
            with st.spinner("Analyzing..."):
                r = predict(fusion_model, mb_model, image, sex, site, personal_hx, family_hx, age, size_mm)

            risk_score = (r['fusion_prob'] + r['mb_prob']) / 2
            is_high_risk = risk_score >= 0.5

            with st.container(border=True):
                top_l, top_r = st.columns([2, 1])
                with top_l:
                    st.markdown('<div class="section-label">Assessment result</div>', unsafe_allow_html=True)
                    pill_class = "risk-high" if is_high_risk else "risk-low"
                    label = "Model output: melanoma class (probability ≥ 0.5)" if is_high_risk else "Model output: non-melanoma class (probability < 0.5)"
                    dot = "●"
                    st.markdown(f'<span class="risk-pill {pill_class}">{dot} {label}</span>', unsafe_allow_html=True)
                with top_r:
                    st.metric("Predicted melanoma probability", f"{risk_score:.0%}")

                st.write("")
                st.progress(risk_score)

                mc1, mc2 = st.columns(2)
                mc1.metric("Fusion: melanoma probability", f"{r['fusion_prob']:.0%}")
                mc2.metric("MetaBlock: melanoma probability", f"{r['mb_prob']:.0%}")

            st.write("")
            with st.expander("Model interpretability details"):
                st.caption(
                    "Estimated contribution of each input (image vs. patient metadata) to this result, "
                    "compared across three methods."
                )
                st.table({
                    "Method": ["Ablation-derived reference (Fusion)", "Integrated Gradients (Fusion)", "MetaBlock gate (mean value)"],
                    "Image contribution": [f"{r['ablation_img']:+.3f}", f"{r['ig_img_frac']:.3f}", "–"],
                    "Metadata contribution": [f"{r['ablation_meta']:+.3f}", f"{1 - r['ig_img_frac']:.3f}", f"{r['gate_val']:.3f}"],
                })
    else:
        with st.container(border=True):
            st.markdown('<div class="section-label">Assessment result</div>', unsafe_allow_html=True)
            st.write("Upload a lesion image and patient details, then click **Analyze**.")

st.write("")
st.markdown(
    '<div class="footnote">Public browser-based research prototype. Not clinically validated; not for clinical use or diagnosis. '
    'Always consult a dermatologist for clinical evaluation.</div>',
    unsafe_allow_html=True,
)
