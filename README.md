# AI-Powered Social Media Caption Generation Engine

This project is a **mood-aware AI caption recommendation engine** that takes an image, generates a description using BLIP, and then uses a powerful LLM (Mistral) to craft **emotionally engaging Instagram-ready captions** with relevant hashtags. Captions are ranked by their **visual-textual similarity using CLIP**, and evaluated for **diversity** using Distinct-n and **uniqueness** using Self-BLEU.

---

##  Features

- 🔍 Extracts rich descriptions from images using **BLIP**
- 💡 Generates 5 unique, engaging captions using **Mistral-7B**
- 📊 Ranks captions based on **CLIP similarity** to the image
- 🧪 Measures **caption diversity** with Distinct-n and **redundancy** using Self-BLEU
- 🌐 Deployed using **Streamlit + Ngrok**

---

##  How to Run (in Google Colab)

1. Upload this repository or ZIP into your Colab environment.
2. Install dependencies:
   ```python
   !pip install -r requirements.txt
   ```
3. Fill in your hugging face token in app.py
   ```python
   token = "your_huggingface_token"
   ```
4. Run the Streamlit app in the background:
   ```python
   !streamlit run app.py &>/content/logs.txt &
   ```
5. Fill in your ngrok token and expose the app:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("your_token_here")
   public_url = ngrok.connect(8501)
   print(public_url)
   ```
6. Go to the URL, upload an image and enjoy 5 mood-rich, hashtagged captions!

---

##  Tech Stack

- `BLIP` – For image-to-text description
- `CLIP` – For image-caption similarity scoring
- `Mistral` – To generate creative, emotionally resonant captions
- `LangChain` – For easy LLM orchestration
- `Streamlit` – UI for uploading images and showing results
- `Pyngrok` – To host the app temporarily on the web
- `nltk` – For Distinct-n and Self-BLEU evaluation

---

##  Folder Structure

```
ai-caption-generator/
├── app.py                  # Main Streamlit app
├── caption_utils.py        # LLM prompt + generation
├── image_utils.py          # BLIP + CLIP models
├── metrics.py              # Diversity and redundancy metrics
├── requirements.txt
└── README.md
```

---

## 📷 Example Use Case

> Upload a beach photo → Get captions like  
> `1. The sound of waves crashing calms my restless mind. #beachpeace #oceanside #mindfulness`

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
