from transformers import GPT2LMHeadModel, GPT2Tokenizer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import torch

def generate_text(prompt):
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    output = model.generate(input_ids, max_length=400, num_return_sequences=1, no_repeat_ngram_size=3)
    
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_pdf(content, image_path, output_path, topic):
    doc = SimpleDocTemplate(output_path, pagesize=letter, leftMargin=30, rightMargin=30, topMargin=50, bottomMargin=50)
    styles = getSampleStyleSheet()
    
    story = []
    
    
    image = Image(image_path, width=300, height=200)
    image_caption_style = ParagraphStyle(name="ImageCaptionStyle", parent=styles["Normal"], fontSize=10, textColor=colors.gray)
    image_caption = Paragraph("Image: Photosynthesis", image_caption_style)
    
    story.append(image)
    story.append(image_caption)
    
    
    story.append(Spacer(1, 10))
    
    
    heading_style = ParagraphStyle(name="HeadingStyle", parent=styles["Normal"], fontSize=18, textColor=colors.black, spaceAfter=10, alignment=1, fontWeight='Bold', textDecoration='underline')
    heading = Paragraph(topic, heading_style)
    story.append(heading)
    

    text = Paragraph(content, styles["Normal"])
    story.append(text)
    
    
    doc.build(story)


topic = "Photosynthesis"
prompt = f"Explain the process of photosynthesis in a detailed and clear manner in one paragraph with a conclusive statement."
generated_content = generate_text(prompt)


image_path = "C:/Users/KIIT/Documents/GitHub/Educational-Content-Generator/synth.jpeg"


output_pdf_path = "output.pdf"


generate_pdf(generated_content, image_path, output_pdf_path, topic)

print("PDF generated successfully.")
