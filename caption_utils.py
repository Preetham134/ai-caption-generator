import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def load_mistral_pipeline():
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    token = "your_huggingface_token"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=torch.float16, device_map="auto")
    text_gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True, return_full_text=False)
    return HuggingFacePipeline(pipeline=text_gen_pipeline)

def generate_mood_caption(description):
    llm = load_mistral_pipeline()
    prompt_template = PromptTemplate(
        template=(
            "You are a social media copywriter.\n\n"
            "Here is a short image description: '{description}'\n\n"
            "Your task is to create exactly 5 short and emotionally engaging captions with a minimum of 8-9 words each based on this description.\n"
            "Each caption should accurately reflect the mood or feeling (like peace, indulgence, adventure, wonder, sadness, anger, or frustration) conveyed in the image description.\n\n"
            "Important guidelines:\n"
            "- Write captions in first person (using 'I', 'my', 'me') or second person (using 'you', 'your') perspective\n"
            "- Avoid using third person pronouns like 'he', 'she', 'his', 'her', 'him', 'them' unless they refer to objects or abstract concepts\n"
            "- Make captions suitable for personal Instagram posts that the user can share as their own\n"
            "- Include 2-3 relevant hashtags that complement each caption and enhance discoverability\n"
            "- Hashtags should be related to the caption's mood, theme, or content\n\n"
            "Respond with a simple numbered list:\n"
            "1. <caption> <hashtags>\n"
            "2. <caption> <hashtags>\n"
            "3. <caption> <hashtags>\n"
            "4. <caption> <hashtags>\n"
            "5. <caption> <hashtags>"
        ),
        input_variables=["description"]
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    output = chain.run(description)
    pattern = r'^\d+\.\s*(.+?)(?=\n\d+\.|$)'
    matches = re.findall(pattern, output, re.MULTILINE)
    captions = [m.strip().strip('"').strip("'") for m in matches if len(m.strip()) > 3]
    return captions[:5]