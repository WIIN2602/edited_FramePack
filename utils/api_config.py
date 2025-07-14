import os
import dotenv

dotenv.load_dotenv(override=True)

# External APIs
GPT_TOKEN = os.getenv("GPT_TOKEN")
LLM_MODEL = os.getenv("LLM_MODEL")
CAPTION_MODEL = os.getenv("CAPTION_MODEL")
FUSING_MODEL = os.getenv("FUSING_MODEL") 
LLM_KEY = os.getenv("LLM_KEY")
BERT_KEY = os.getenv("BERT_KEY")
BOTNOI_TOKEN = os.getenv("BOTNOI_TOKEN")
BOT_URL = os.getenv("BOT_URL")
MODEL_ASR = os.getenv("MODEL_ASR")
TRANSCRIBE_URL = os.getenv("TRANSCRIBE_URL")
TRANSCRIBE2_URL = os.getenv("TRANSCRIBE2_URL")

# Physical Storage

SERVICE_DIR = os.getenv("SERVICE_DIR")
VID_SRC_DIR = os.getenv("VID_SRC_DIR")
VID_SCN_DIR = os.getenv("VID_SCN_DIR")
VID_SUM_DIR = os.getenv("VID_SUM_DIR")
CACHE_DIR = os.getenv("CACHE_DIR")
SCRIPT_DIR = os.getenv("SCRIPT_DIR")
VOICE_DIR = os.getenv("VOICE_DIR")
IMAGE_DIR = os.getenv("IMAGE_DIR")

# Create Directories
if not os.path.exists(SERVICE_DIR):
    os.makedirs(SERVICE_DIR, exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, VID_SRC_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, VID_SRC_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, VID_SCN_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, VID_SCN_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, VID_SUM_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, VID_SUM_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, CACHE_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, CACHE_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, SCRIPT_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, SCRIPT_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, VOICE_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, VOICE_DIR), exist_ok=True)
if not os.path.exists(os.path.join(SERVICE_DIR, IMAGE_DIR)):
    os.makedirs(os.path.join(SERVICE_DIR, IMAGE_DIR), exist_ok=True)