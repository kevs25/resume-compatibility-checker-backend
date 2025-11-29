from fastapi import UploadFile
import PyPDF2
import io

async def parse_pdf(file: UploadFile) -> str:
    """
    Parse PDF file and extract text content
    
    Args:
        file: UploadFile object containing PDF
    
    Returns:
        Extracted text from PDF
    """
    try:
        # Read file content
        content = await file.read()
        
        # Create a PDF reader object
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        # Clean up the text
        text = text.strip()
        
        if not text:
            raise ValueError("No text could be extracted from the PDF")
        # print("pdf text: ", text)
        return text
    
    except Exception as e:
        raise Exception(f"Error parsing PDF: {str(e)}")
    
    finally:
        # Reset file pointer for potential reuse
        await file.seek(0)