import streamlit as st
from PIL import Image
from image_utils import get_blip_description, generate_image_features, compute_caption_similarities
from caption_utils import generate_mood_caption
from metrics import distinct_n, self_bleu

st.markdown("<h1 style='text-align: center;'>AI-Powered Social Media Caption Generation Engine</h1>", unsafe_allow_html=True)
st.markdown("Upload an image and get 5 social-ready captions that capture the emotion ✨", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🧠 Thinking up some beautiful and engaging captions..."):
        description = get_blip_description(image)
        captions = generate_mood_caption(description)
        image_features, clip_model, clip_processor = generate_image_features(image)
        similarities = compute_caption_similarities(captions, image_features, clip_model, clip_processor)

        # ---- Compute metrics silently ----
        d_score = distinct_n(captions, n=2)
        s_bleu = self_bleu(captions)

    scored = sorted(zip(captions, similarities), key=lambda x: x[1], reverse=True)

    st.markdown("---")
    st.subheader("💬 Your Captions")
    st.markdown("Captions Ranked by Relevance to Your Image:")

    
    for i, (cap, sim) in enumerate(scored, 1):
        st.markdown(
            f"""
            <div style="border:1px solid #ddd; padding:12px; margin:10px 0; border-radius:10px; background-color:#fafafa;">
                <strong>{i}. {cap}</strong><br>
                <span style="color:#888;">Similarity Score: <code>{sim:.2f}</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Use these markdown options when you want to see the metrics in the UI
    # st.markdown("---")
    # st.markdown(f"**Diversity (Distinct-2):** `{d_score:.2f}`")
    # st.markdown(f"**Self-BLEU Score:** `{s_bleu:.2f}`")