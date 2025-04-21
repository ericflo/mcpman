import json
import logging
from typing import Dict, List, Any, Optional

import httpx


class LLMClient:
    """
    Manages communication with the LLM provider.
    
    Handles:
    - API requests to LLM providers
    - Response parsing
    - Tool schema formatting
    """

    def __init__(self, api_key: str, api_url: str, model_name: str) -> None:
        """
        Initialize an LLMClient instance.
        
        Args:
            api_key: API key for the LLM provider
            api_url: API URL for the LLM provider
            model_name: Name of the model to use
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name

    def get_response(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get a response message object from the LLM.
        
        Args:
            messages: List of message objects to send to the LLM
            tools: Optional list of tool definitions
            temperature: Sampling temperature for the LLM
            max_tokens: Maximum number of tokens for the response
            
        Returns:
            Response message object from the LLM
        """
        logging.debug(
            f"Sending {len(messages)} messages to LLM. {'Including tools.' if tools else 'No tools.'}"
        )

        # Prepare headers and payload
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "messages": messages,
            "model": self.model_name,
            "temperature": temperature,
        }
        
        # Add optional parameters
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        if tools:
            payload["tools"] = tools

        try:
            # Make the API request
            with httpx.Client(timeout=60.0) as client:
                # Only log full payload on DEBUG level
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    try:
                        payload_str = json.dumps(payload, indent=2)
                        logging.debug(f"Sending LLM Payload:\n{payload_str}")
                    except Exception as e:
                        logging.debug(
                            f"Could not serialize LLM payload for logging: {e}"
                        )
                else:
                    logging.info(
                        f"Sending request to LLM model {self.model_name} at {self.api_url}..."
                    )

                # Send the request
                response = client.post(self.api_url, headers=headers, json=payload)
                logging.debug(
                    f"Received response from LLM: Status {response.status_code}"
                )
                response.raise_for_status()

                # Parse the response
                data = response.json()
                if not data or "choices" not in data or not data["choices"]:
                    raise KeyError("Invalid response: 'choices' missing or empty.")
                if "message" not in data["choices"][0]:
                    raise KeyError(
                        "Invalid response: 'message' missing in first choice."
                    )

                message = data["choices"][0]["message"]
                logging.debug(f"LLM response message object: {message}")
                return message

        except httpx.RequestError as e:
            logging.error(f"Error communicating with LLM at {self.api_url}: {e}")
            # Return an error-like message object
            return {"role": "assistant", "content": f"Error: Could not reach LLM: {e}"}
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logging.error(f"Error parsing LLM response: {e}")
            return {
                "role": "assistant",
                "content": f"Error: Invalid LLM response format: {e}",
            }


def create_llm_client(
    provider_config: Dict[str, str],
    provider_name: Optional[str] = None,
) -> LLMClient:
    """
    Create an LLM client based on provider configuration.
    
    Args:
        provider_config: Provider configuration with url, key, and model
        provider_name: Optional provider name for logging
        
    Returns:
        Configured LLMClient instance
    """
    # Log the LLM configuration
    provider_str = provider_name or "custom"
    logging.info(f"Initializing LLM client for provider: {provider_str}")
    logging.info(f"Model: {provider_config['model']}")
    logging.info(f"API URL: {provider_config['url']}")
    
    # Create and return the client
    return LLMClient(
        api_key=provider_config["key"],
        api_url=provider_config["url"],
        model_name=provider_config["model"],
    )