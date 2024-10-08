import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import PyPDF2 as pdf
from dotenv import load_dotenv
import torch

# Load environment variables
load_dotenv()

# Load the GPT-2 model and tokenizer (using the smaller, free model)
model_name = "gpt2"  # Use a free model if "gpt2-xl" causes issues

# Ensure PyTorch is available
if not torch.cuda.is_available() and torch.version.__version__ == '':
    st.error("PyTorch is not installed or not correctly set up. Please install PyTorch and restart the runtime.")
    st.stop()

try:
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
except ImportError as e:
    st.error(f"An import error occurred: {e}")
    st.stop()

def get_gpt2_response(prompt):
    try:
        # Tokenize the input prompt with truncation to fit model constraints
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        # Generate text using the model
        outputs = model.generate(
            **inputs,
            max_length=1500,  # Increase this as needed but be mindful of memory
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        # Decode the generated text
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except IndexError as e:
        return f"An index error occurred: {str(e)}"
    except ValueError as e:
        return f"A value error occurred: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# General prompt for ATS evaluation
input_prompt_ats = """ 
Bonjour, agissez comme un ATS (Système de Suivi des Candidatures) expérimenté 
ou très compétent avec une compréhension approfondie de tous les domaines. Votre tâche est d'évaluer ce CV en fonction de la description du poste fournie. Considérez que le marché de l'emploi 
est très compétitif et fournissez la meilleure assistance pour améliorer les CV. 
Attribuez un pourcentage de correspondance basé sur la description du poste et 
les mots-clés manquants avec une grande précision. Fournissez également un résumé 
des points forts du CV et des domaines à améliorer.

cv:{text}
description:{jd}

Je souhaite que la réponse soit en trois sections distinctes comme suit:
1. **Correspondance JD**: Un pourcentage indiquant dans quelle mesure le CV correspond à la description du poste.
2. **MotsClésManquants**: Une liste de mots-clés importants qui sont absents du CV.
3. **RésuméProfil**: Un résumé des points forts du CV et des domaines à améliorer.

Répondez en utilisant cette structure pour chaque section:
1. **Correspondance JD**: [Pourcentage]
2. **MotsClésManquants**: [Liste de mots-clés]
3. **RésuméProfil**: [Résumé du profil]
"""

# Streamlit app setup
st.title("ATS Intelligent")
st.text("Améliorez votre CV")

# Option to choose CV type
cv_type = st.radio("Choisissez le type de CV", ("ATS CV", "Normal CV"))

jd = st.text_area("Collez la Description du Poste")
uploaded_file = st.file_uploader("Téléchargez Votre CV", type="pdf", help="Veuillez télécharger le pdf")

if st.button("Soumettre"):
    if uploaded_file is not None:
        with st.spinner('Traitement en cours...'):
            # Read and process PDF
            text = input_pdf_text(uploaded_file)
            # Generate response from GPT-2 model
            prompt = input_prompt_ats.format(text=text, jd=jd)
            response = get_gpt2_response(prompt)
            st.subheader(response)
    else:
        st.warning("Veuillez télécharger un CV au format PDF.")
