
![Screenshot from 2025-05-22 18-46-23](https://github.com/user-attachments/assets/373a99fe-c1e2-4721-96de-c38d4c3e6b0d)

## Gemini Cloud Speech-to-text (Home Assistant Integration)
This is a competitor of Google Cloud Speech-to-text. It utilizes the latest gemini LLM to automatically understand and transcribe your speech which is 100% more reliable than traditional neural network STT architecture. And the best part is it is **free**. Get your api_key at [Google AI Studio](https://aistudio.google.com/app/apikey) now!


## Installation
#### HACS (Recommended)
- HACS > Integrations > 3 dots (upper top corner) > Custom repositories > URL: https://github.com/Steven-Low/gemini_cloud_stt, Category: Integration > Add > wait > Gemini Cloud STT > Install

#### Manual
- Clone the repo of the master branch to `/config/custom_components` in HA instance.
```
git clone https://github.com/Steven-Low/gemini_cloud_stt.git
```

### Important Notes
The language configuration for the transcription is **not** done on the Voice Assistant setup (It is completely ignore there). You shall set your preferences (model, language, prompt) at the integration configure section.

##### Intra-lingual
The **default** transcibe method will be intra-lingual where the speech to text will be same language
- English speech --> English text
- Japanese speech --> Japanese text
##### Inter-lingual
If you select the output language, it will auto convert your speech to the selected language.
- Any speech --> selected language
##### Free-4-all
Play with the prompt/value_template for advanced use cases. no warranty :D

