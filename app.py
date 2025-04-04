import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from PyPDF2 import PdfReader
import langdetect
import io

# Configure page
st.set_page_config(page_title="Exam Question Predictor", page_icon="📚")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def detect_language(text):
    """Detect the language of the input text"""
    try:
        lang = langdetect.detect(text)
        return lang
    except:
        return 'en'  # default to English if detection fails

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Simple title
st.title("Exam Question Predictor")
st.write("Upload your previous exams and lesson content as PDFs to generate potential questions.")

try:
    tokenizer, model = load_model()
    
    # File upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Previous Exams")
        previous_exams_file = st.file_uploader(
            "Upload previous exam questions (PDF)",
            type="pdf",
            help="Upload a PDF file containing previous exam questions"
        )

    with col2:
        st.subheader("Lesson Content")
        lesson_content_file = st.file_uploader(
            "Upload lesson content (PDF)",
            type="pdf",
            help="Upload a PDF file containing the lesson content"
        )

    if previous_exams_file and lesson_content_file:
        # Extract text from PDFs
        previous_exams = extract_text_from_pdf(previous_exams_file)
        lesson_content = extract_text_from_pdf(lesson_content_file)
        
        # Detect language from previous exams
        doc_language = detect_language(previous_exams)
        
        # Show extracted text (collapsible)
        with st.expander("View Extracted Content"):
            st.write("Previous Exams Content:")
            st.text(previous_exams[:500] + "..." if len(previous_exams) > 500 else previous_exams)
            st.write("\nLesson Content:")
            st.text(lesson_content[:500] + "..." if len(lesson_content) > 500 else lesson_content)

        if st.button("Generate Exam Questions"):
            with st.spinner("Generating questions..."):
                # Create prompt based on detected language
                if doc_language == 'fr':
                    prompt_template = f"""
                    Tâche : Générer 5 questions d'examen basées sur :
                    Style des examens précédents : {previous_exams}
                    Contenu de la leçon : {lesson_content}
                    
                    Générer 5 questions d'examen qui correspondent au style précédent et testent le contenu actuel.
                    """
                elif doc_language == 'es':
                    prompt_template = f"""
                    Tarea: Generar 5 preguntas de examen basadas en:
                    Estilo de exámenes anteriores: {previous_exams}
                    Contenido de la lección: {lesson_content}
                    
                    Generar 5 preguntas de examen que coincidan con el estilo anterior y evalúen el contenido actual.
                    """
                else:  # default to English
                    prompt_template = f"""
                    Task: Generate 5 exam questions based on:
                    Previous exam style: {previous_exams}
                    Current lesson content: {lesson_content}
                    
                    Generate 5 exam questions that match the previous style and test the current content.
                    """
                
                # Generate questions
                inputs = tokenizer(prompt_template, return_tensors="pt", max_length=1024, truncation=True)
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    num_beams=4
                )
                
                predicted_questions = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Display results
                st.success("Questions Generated!")
                st.write("Potential Exam Questions:")
                
                # Format the questions
                questions = predicted_questions.split("\n")
                for i, question in enumerate(questions, 1):
                    if question.strip():
                        st.write(f"{i}. {question.strip()}")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Try refreshing the page if the model fails to load.")

st.write("---")
st.write("Made with Streamlit and Hugging Face Transformers")
