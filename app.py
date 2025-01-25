import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader  # Library to read PDF files

nltk.download('punkt')
nltk.download('stopwords')

# Loading models
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
clf = pickle.load(open('cls.pkl', 'rb'))

def cleanResume(txt):
    cleanText = re.sub('http\S+', ' ', txt)  # Removing URLs
    cleanText = re.sub('RT|cc', ' ', cleanText)  # Removing retweets and 'cc'
    cleanText = re.sub('#\S+', ' ', cleanText)  # Removing hashtags
    cleanText = re.sub('@\S+', ' ', cleanText)  # Removing mentions
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)  # Removing punctuation
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)  # Removing non-ASCII characters
    cleanText = re.sub('\s+', ' ', cleanText)  # Removing extra whitespaces
    return cleanText

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Web app
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload your resume", type=['txt', 'docx', 'pdf'])
    
    if upload_file is not None:
        try:
            if upload_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(upload_file)
            else:
                resume_bytes = upload_file.read()
                resume_text = resume_bytes.decode('utf-8')  # For txt/docx files
                
        except Exception as e:
            st.error("Error reading the file. Please ensure the file format is correct.")
            st.error(str(e))
            return

        # Clean the resume text
        cleaned_resume = cleanResume(resume_text)

        # Transform and predict
        tfidf_resume = tfidf.transform([cleaned_resume])
        prediction_id = clf.predict(tfidf_resume)[0]

        # Map prediction to category
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: 'Python/ Developer',
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and Fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "Dotnet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, 'Unknown')
        st.write("Resume Profile is: ", category_name)

if __name__ == "__main__":
    main()
