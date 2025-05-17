"""Config flow for Gemini Cloud STT integration."""

from __future__ import annotations
 
from typing import Any

import logging

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry, ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.const import CONF_MODEL, CONF_API_KEY
from homeassistant.core import callback, HomeAssistant
 
from google import genai
from .const import (
    DEFAULT_MODEL,
    DOMAIN,
    SUPPORTED_MODELS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
    }
)

OPTIONS_SCHEMA = vol.Schema(
    {
        vol.Optional(CONF_MODEL, default=DEFAULT_MODEL): vol.In(SUPPORTED_MODELS),
    }
)

async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input for gemini client."""
    def setup_gemini_client():
        client = genai.Client(api_key=data[CONF_API_KEY])
    await hass.async_add_executor_job(setup_gemini_client)

class GeminiCloudConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Gemini Cloud STT integration."""

    VERSION = 1

    _name: str | None = None

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try: 
            await validate_input(self.hass, user_input)
        except Exception as err:
            errors["base"] = "unknown"
            _LOGGER.info("Error validating gemini api key: ", err)
        else:
            return self.async_create_entry(
                title="Gemini Cloud STT",
                data=user_input,
            )
        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

 
    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return GeminiCloudOptionsFlowHandler(config_entry)
    

class GeminiCloudOptionsFlowHandler(OptionsFlow):
    """Handle a options flow for Google Cloud STT integration."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize Google Cloud STT options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="Gemini Cloud STT", data=user_input)
    
        return self.async_show_form(step_id="init", data_schema=OPTIONS_SCHEMA)