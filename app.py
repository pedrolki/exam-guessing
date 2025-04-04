import streamlit as st
from transformers import AutoModelForSeq2SeqGeneration, AutoTokenizer
import torch

# Configure page
st.set_page_config(page_title="Exam Question Predictor", page_icon="📚")

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
    return tokenizer, model

# Simple title
st.title("Exam Question Predictor")
st.write("Generate potential exam questions based on previous exams and lesson content.")

try:
    tokenizer, model = load_model()
    
    # Input sections
    previous_exams = st.text_area(
        "Previous Exam Questions",
        placeholder="Enter previous exam questions here...",
    )

    lesson_content = st.text_area(
        "Lesson Content",
        placeholder="Enter current lesson content here...",
    )

    if st.button("Generate Exam Questions"):
        if not previous_exams or not lesson_content:
            st.error("Please fill in both fields")
        else:
            with st.spinner("Generating questions..."):
                prompt = f"""
                Task: Generate 5 exam questions based on:
                Previous exam style: {previous_exams}
                Current lesson content: {lesson_content}
                
                Generate 5 exam questions that match the previous style and test the current content.
                """
                
                # Generate questions
                inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
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
