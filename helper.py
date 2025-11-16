import google.generativeai as genai
import PyPDF2 as pdf
import json

def configure_genai(api_key):
    """Comfigure the Generative AI API with error handling."""
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        raise Exception(f"Failed to configure Generative AI :{str(e)}")
    
def get_gemini_response(prompt):    
    """
    Generate a response using Gemini with enhanced error handling and response validation.
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        # Ensure response is not empty
        if not response or not response.text:
            raise Exception("Empty response received from Gemini")

        # Try to parse the response as JSON
        try:
            response_json = json.loads(response.text)

            # Validate required fields
            required_fields = ["JD Match", "MissingKeywords", "Profile Summary"]
            for field in required_fields:
                if field not in response_json:
                    raise ValueError(f"Missing required field: {field}")

            return response.text
        
        except json.JSONDecodeError:
            # Agar Gemini ka response JSON me parse na ho paye
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern,response.text,re.DOTALL)
            if match:
                return match.group()
            else:
                raise Exception('could not extract valid JSON response')


    except Exception as e:
        raise Exception(f"Error generating Gemini response: {str(e)}")

def extract_pdf_text(uploaded_file):
    """Extract text from pdf with enhanced error handling."""
    try:
        reader = pdf.PdfReader(uploaded_file)
        if len(reader.pages)==0:
            raise Exception("PDF file is empty")
        
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        
        if not text :
            raise Exception("No text could be extracted from PDF")
        
        return "".join(text)
    
    except Exception as e:
        raise Exception(f'error extracting PDF text: {str(e)}')

def prepare_prompt(resume_text, job_description):
    """ Prepare the input prompt with improved structure and validation."""

    if not resume_text or not job_description:
        raise Exception("resume text and job description cannot be empty")
    
    prompt_template = """
    Act as an expert ATS (Applicant Tracking System) specialist with deep expertise in:
    - Technical fields
    - Software engineering
    - Data science
    - Data analysis
    - Big data engineering

    Evaluate the following resume against the job description. 
    Consider that the job market is highly competitive. 
    Provide detailed feedback for resume improvement.

    Resume:
    {resume_text}

    Job Description:
    {job_description}


    Provide a response in the following JSON format Only:
    {{
    "JD Match":"percentage between 1-100",
    "MissingKeywords": ["Keyword1","keyword2", ...]
    "Profile Summary": "detailed analysis of the match and specific improvement sugggestions"
    }}
    """

    return prompt_template.format(
        resume_text=resume_text.strip(),
        job_description=job_description.strip()
    )