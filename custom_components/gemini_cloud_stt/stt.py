"""Support for the Google Cloud speech to text service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
import io
import wave
from google import genai
from google.genai import types

from homeassistant.components import stt
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_MODEL, CONF_API_KEY, CONF_LANGUAGE, CONF_VALUE_TEMPLATE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback


from .const import CONF_PROMPT, DEFAULT_MODEL, SAMPLE_CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, SUPPORTED_LANGUAGES
import logging

_LOGGER = logging.getLogger(__name__)




async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Gemini Cloud speech-to-text."""

    async_add_entities(
        [
            GeminiCloudSTTProvider(
                hass,
                config_entry.data[CONF_API_KEY],
                config_entry.options.get(CONF_MODEL, DEFAULT_MODEL),
                config_entry.options.get(CONF_LANGUAGE, "auto"),
                config_entry.options.get(CONF_VALUE_TEMPLATE, CONF_PROMPT)
            ),
        ]
    )


class GeminiCloudSTTProvider(stt.SpeechToTextEntity):
    """The Gemini Cloud STT API provider."""

    _attr_name = "Gemini Cloud"
    _attr_unique_id = "gemini-cloud-speech-to-text"

    def __init__(self, hass, api_key, model, language, prompt) -> None:
        """Init Gemini Cloud STT service."""
        self.hass = hass

        self._model = model
        self._api_key = api_key
        self._language = language
        self._prompt = prompt
        self._client = None

    @property
    def supported_languages(self) -> list[str]:
        """Return a list of supported languages."""
        return SUPPORTED_LANGUAGES

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    @staticmethod
    async def convert_raw_to_wav(audio_data: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(SAMPLE_CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data)
        return buffer.getvalue()


    def setup_genai_client(self):
        self._client = genai.Client(api_key=self._api_key)

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        """Process an audio stream to STT service."""

        if self._client is None:
            await self.hass.async_add_executor_job(self.setup_genai_client)

        # Collect data
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        wav_data = await self.convert_raw_to_wav(audio_data)


        def job():
            _LOGGER.info("PROMPT: " + self._prompt if self._language == "auto" else f"{CONF_PROMPT} to {self._language}")
            return self._client.models.generate_content(
                model=self._model,
                contents=[
                    self._prompt if self._language == "auto" else f"{CONF_PROMPT} to {self._language}",
                    types.Part.from_bytes(
                        data=wav_data,
                        mime_type='audio/wav',
                    )
                ]
            )

        async with asyncio.timeout(10):
            assert self.hass
            response = await self.hass.async_add_executor_job(job)
            if response.text:
                _LOGGER.info(response.text)
                return SpeechResult(
                    response.text,
                    SpeechResultState.SUCCESS,
                )
            return SpeechResult(None, SpeechResultState.ERROR)