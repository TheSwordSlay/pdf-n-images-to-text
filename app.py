import os
import io
import base64
import logging
import re
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
import requests
from PIL import Image
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
import tempfile

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

class PDFProcessor:
    def __init__(self):
        self.openrouter_headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def extract_content_with_positions(self, pdf_path):
        """Extract text and images with their positions from PDF"""
        try:
            doc = fitz.open(pdf_path)
            extracted_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_content = {
                    'page_number': page_num + 1,
                    'page_size': page.rect,
                    'text_blocks': [],
                    'images': []
                }
                
                # Extract text blocks with positions
                text_dict = page.get_text("dict")
                for block in text_dict["blocks"]:
                    if "lines" in block:  # Text block
                        block_text = ""
                        for line in block["lines"]:
                            for span in line["spans"]:
                                block_text += span["text"]
                            block_text += "\n"
                        
                        if block_text.strip():
                            page_content['text_blocks'].append({
                                'text': block_text.strip(),
                                'bbox': block["bbox"],  # (x0, y0, x1, y1)
                                'font_info': line["spans"][0] if line.get("spans") else {}
                            })
                
                # Extract images with positions
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_data).decode()
                            
                            # Get image position
                            img_rects = page.get_image_rects(xref)
                            bbox = img_rects[0] if img_rects else fitz.Rect(0, 0, 100, 100)
                            
                            page_content['images'].append({
                                'index': img_index,
                                'base64': img_base64,
                                'bbox': bbox,
                                'description': None  # Will be filled by OpenRouter
                            })
                        
                        pix = None
                    except Exception as e:
                        logger.error(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                
                extracted_content.append(page_content)
            
            doc.close()
            return extracted_content
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    
    def describe_image_with_openrouter(self, image_base64, language='english'):
        """Use OpenRouter API with Gemini 2.5 Flash to describe image"""
        try:
            # Create dynamic prompt based on detected language
            prompt_text = f"""You are an AI assistant analyzing an image from a document. Your task is to provide a replacement text description. First, determine the image's primary function and then describe it appropriately.

- If the image is a **photograph, chart, diagram, table, graph, or other things that conveys substantive information** , provide a detailed description focusing on the key message, data, and subjects shown.
- If the image is a **logo**, simply identify it concisely, like "Logo of [Company Name]".
- If the image is a **signature**, state that it is a signature. If the name is legible, include it. For example, "Signature of Jane Doe" or "A signature".
- If the image is purely **decorative** (e.g., a border, flourish, or abstract graphic), provide a very brief, functional description like "Decorative border".

Analyze the image's importance and function before describing it. Respond entirely in {language}."""
            
            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=self.openrouter_headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                return "Error: Could not describe image"
                
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            return "Error: Could not describe image"
    
    def create_pdf_with_text_replacements(self, content, output_path):
        """Create a new PDF with images replaced by text descriptions"""
        try:
            # Create a new PDF document
            doc = SimpleDocTemplate(output_path, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Create custom styles
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=10,
                leading=12,
                alignment=TA_JUSTIFY,
                spaceAfter=6
            )
            
            image_replacement_style = ParagraphStyle(
                'ImageReplacement',
                parent=styles['Normal'],
                fontSize=9,
                leading=11,
                alignment=TA_LEFT,
                spaceAfter=12,
                spaceBefore=6,
                leftIndent=20,
                rightIndent=20,
                borderWidth=1,
                borderPadding=8
            )
            
            for page_content in content:
                # # Add page header
                # page_header = f"--- Page {page_content['page_number']} ---"
                # story.append(Paragraph(page_header, styles['Heading2']))
                # story.append(Spacer(1, 12))
                
                # Combine text blocks and image descriptions, sorted by position
                all_elements = []
                
                # Add text blocks
                for text_block in page_content['text_blocks']:
                    all_elements.append({
                        'type': 'text',
                        'content': text_block['text'],
                        'y_pos': text_block['bbox'][1],  # y0 coordinate
                        'bbox': text_block['bbox']
                    })
                
                # Add image descriptions
                for image in page_content['images']:
                    if image['description']:
                        all_elements.append({
                            'type': 'image_description',
                            'content': f"[IMAGE DESCRIPTION: {image['description']}]",
                            'y_pos': image['bbox'].y0,  # y0 coordinate
                            'bbox': image['bbox']
                        })
                
                # Sort elements by vertical position (top to bottom)
                all_elements.sort(key=lambda x: x['y_pos'])
                
                # Add elements to story
                for element in all_elements:
                    if element['type'] == 'text':
                        if element['content'].strip():
                            story.append(Paragraph(element['content'], normal_style))
                    elif element['type'] == 'image_description':
                        story.append(Paragraph(element['content'], image_replacement_style))
                
                # Add page break except for the last page
                if page_content != content[-1]:
                    story.append(PageBreak())
            
            # Build the PDF
            doc.build(story)
            logger.info(f"PDF created successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating PDF: {str(e)}")
            return False
    
    def process_pdf(self, pdf_path):
        """Main processing function - now returns a new PDF file"""
        try:
            # Extract text and images with positions
            logger.info("Extracting text and images from PDF...")
            content = self.extract_content_with_positions(pdf_path)
            
            # Detect document language
            logger.info("Detecting document language...")
            detected_language = self.detect_document_language_from_blocks(content)
            logger.info(f"Detected language: {detected_language}")
            
            # Process images with OpenRouter
            logger.info("Processing images with OpenRouter API...")
            for page in content:
                for image in page['images']:
                    if image['base64']:
                        description = self.describe_image_with_openrouter(image['base64'], detected_language)
                        image['description'] = description
                        # Remove base64 data to reduce memory usage
                        image.pop('base64', None)
            
            # Create output PDF with text replacements
            logger.info("Creating new PDF with image descriptions...")
            output_path = tempfile.mktemp(suffix='.pdf')
            success = self.create_pdf_with_text_replacements(content, output_path)
            
            if success:
                return {
                    'success': True,
                    'output_path': output_path,
                    'detected_language': detected_language
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to create output PDF'
                }
            
        except Exception as e:
            logger.error(f"Error in process_pdf: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def detect_document_language_from_blocks(self, content):
        """Detect language from text blocks"""
        try:
            # Combine text from first few pages
            sample_text = ""
            pages_to_sample = min(3, len(content))
            
            for i in range(pages_to_sample):
                for text_block in content[i]['text_blocks'][:5]:  # First 5 blocks per page
                    sample_text += text_block['text'][:200] + " "  # First 200 chars per block
            
            # Truncate if too long
            if len(sample_text) > 1500:
                sample_text = sample_text[:1500]
            
            if not sample_text.strip():
                return "english"  # Default fallback
            
            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": f"""Analyze the following text and identify the primary language. Respond with ONLY the language name in English (e.g., "english", "indonesian", "spanish", "french", "german", "chinese", "japanese", etc.).

Text to analyze:
{sample_text}

Language:"""
                    }
                ],
                "max_tokens": 50,
                "temperature": 0.1
            }
            
            response = requests.post(
                OPENROUTER_BASE_URL,
                headers=self.openrouter_headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_language = result['choices'][0]['message']['content'].strip().lower()
                logger.info(f"LLM detected language: {detected_language}")
                return detected_language
            else:
                logger.error(f"Language detection API error: {response.status_code}")
                return "english"  # Fallback
                
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return "english"  # Fallback

# Initialize processor
pdf_processor = PDFProcessor()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'PDF Image Replacement API - Replaces images in PDFs with descriptive text',
        'version': '2.0.0',
        'endpoints': {
            '/convert': 'POST - Upload PDF, returns PDF with images replaced by text descriptions',
            '/convert-json': 'POST - Upload PDF, returns JSON with extracted text and image descriptions'
        }
    })

@app.route('/convert', methods=['POST'])
def convert_pdf():
    """Convert PDF with images replaced by text descriptions"""
    try:
        # Check if API key is configured
        if not OPENROUTER_API_KEY:
            return jsonify({
                'success': False,
                'error': 'OpenRouter API key not configured'
            }), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                'success': False,
                'error': 'Only PDF files are allowed'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_input_path = f"/tmp/input_{filename}"
        file.save(temp_input_path)
        
        try:
            # Process the PDF
            result = pdf_processor.process_pdf(temp_input_path)
            
            # Clean up input temporary file
            os.remove(temp_input_path)
            
            if result['success']:
                # Generate output filename
                base_name = os.path.splitext(filename)[0]
                output_filename = f"{base_name}.pdf"
                
                # Return the processed PDF file
                return send_file(
                    result['output_path'],
                    as_attachment=True,
                    download_name=output_filename,
                    mimetype='application/pdf'
                )
            else:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 500
            
        except Exception as e:
            # Clean up temporary files in case of error
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in convert_pdf: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/convert-json', methods=['POST'])
def convert_pdf_json():
    """Convert PDF and return JSON with text and image descriptions (legacy endpoint)"""
    try:
        # Check if API key is configured
        if not OPENROUTER_API_KEY:
            return jsonify({
                'success': False,
                'error': 'OpenRouter API key not configured'
            }), 500
        
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({
                'success': False,
                'error': 'Only PDF files are allowed'
            }), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_input_path = f"/tmp/input_{filename}"
        file.save(temp_input_path)
        
        try:
            # Extract content for JSON response
            content = pdf_processor.extract_content_with_positions(temp_input_path)
            detected_language = pdf_processor.detect_document_language_from_blocks(content)
            
            # Process images with OpenRouter
            for page in content:
                for image in page['images']:
                    if image['base64']:
                        description = pdf_processor.describe_image_with_openrouter(image['base64'], detected_language)
                        image['description'] = description
                        # Remove base64 data to reduce response size
                        image.pop('base64', None)
            
            # Clean up input temporary file
            os.remove(temp_input_path)
            
            # Create merged text content
            merged_text = ""
            for page in content:
                merged_text += f"\n--- Page {page['page_number']} ---\n"
                
                # Combine text blocks and image descriptions
                all_elements = []
                for text_block in page['text_blocks']:
                    all_elements.append({
                        'type': 'text',
                        'content': text_block['text'],
                        'y_pos': text_block['bbox'][1]
                    })
                
                for image in page['images']:
                    if image['description']:
                        all_elements.append({
                            'type': 'image_description',
                            'content': f"[IMAGE: {image['description']}]",
                            'y_pos': image['bbox'].y0
                        })
                
                # Sort by position and add to merged text
                all_elements.sort(key=lambda x: x['y_pos'])
                for element in all_elements:
                    merged_text += element['content'] + "\n"
                
                merged_text += "\n" + "="*50 + "\n"
            
            return jsonify({
                'success': True,
                'detected_language': detected_language,
                'merged_text': merged_text,
                'page_details': content
            })
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_input_path):
                os.remove(temp_input_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error in convert_pdf_json: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)