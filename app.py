import streamlit as st
import google.generativeai as genai
from datetime import datetime

# Configure page
st.set_page_config(page_title="Exam Prediction AI", page_icon="📚")

# Initialize Gemini-pro
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-pro')

# Add custom CSS
st.markdown("""
    <style>
    .stTextArea textarea {
        height: 200px;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📚 Exam Prediction AI")
st.markdown("Predict potential exam questions based on previous exams and lesson content.")

# Input sections
with st.container():
    previous_exams = st.text_area(
        "Previous Exam Questions",
        placeholder="Enter previous exam questions here...",
        help="Enter questions from previous exams to help the AI understand the professor's style"
    )

    lesson_content = st.text_area(
        "Lesson Content",
        placeholder="Enter current lesson content here...",
        help="Enter the content of the current lessons that might be tested"
    )

    if st.button("🔮 Predict Next Exam Questions", use_container_width=True):
        if not previous_exams or not lesson_content:
            st.error("Please fill in both fields")
        else:
            with st.spinner("Generating predictions..."):
                prompt = f"""
                Based on the following previous exam questions and lesson content, predict possible questions for the next exam.
                
                Previous Exam Questions:
                {previous_exams}
                
                Lesson Content:
                {lesson_content}
                
                Generate 5 potential exam questions that could appear on the next exam. Format them as a numbered list.
                For each question:
                1. Make it similar in style to the previous questions
                2. Ensure it tests understanding of the lesson content
                3. Match the difficulty level of previous questions
                
                Return ONLY the questions, numbered 1-5.
                """
                
                try:
                    response = model.generate_content(prompt)
                    
                    st.success("✨ Predicted Questions Generated!")
                    st.markdown("### 📝 Potential Exam Questions:")
                    st.markdown(response.text)
                    
                    # Add timestamp
                    st.caption(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("*Note: This is an AI prediction tool. The actual exam questions may vary.*")
