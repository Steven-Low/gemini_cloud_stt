"""Constants for the Google Cloud Speech-to-Text integration."""


DOMAIN = "gemini_cloud_stt"
DEFAULT_MODEL = "gemini-2.0-flash"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
SUPPORTED_MODELS = [
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview-native-audio-dialog",
    "gemini-2.5-pro-preview-05-06",
]

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
SAMPLE_CHANNELS = 1