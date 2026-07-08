from dotenv import load_dotenv
import os
load_dotenv()
print('GROQ_API_KEY set:', bool(os.environ.get('GROQ_API_KEY', '').strip()))
print('GROQ_API_KEY len:', len(os.environ.get('GROQ_API_KEY', '')))
from app.generator import check_provider_status
print(check_provider_status('groq'))
