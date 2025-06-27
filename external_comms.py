# --- START OF FILE external_comms.py ---

import asyncio
import logging
import time
import random
import threading # <<< ADDED IMPORT
from typing import Any, Dict, List, Tuple, Optional
import httpx # Make sure httpx is imported
import json


# Assuming necessary libraries (ollama, chromadb) are available
# and configuration is imported where needed.
try: import ollama; OLLAMA_AVAILABLE = True
except ImportError: ollama = None; OLLAMA_AVAILABLE = False; logging.error("external_comms: ollama missing.")
try: import chromadb; CHROMADB_AVAILABLE = True
except ImportError: chromadb = None; CHROMADB_AVAILABLE = False; logging.error("external_comms: chromadb missing.")

# Import constants and potentially logging setup if needed directly
# from . import agent_config # Use relative import if available
_agent_config_ext = globals().get('agent_config') # Get from global scope


logger_external_comms = logging.getLogger(__name__ + ".external_comms")
OLLAMA_API_BASE_URL = "http://127.0.0.1:11434" # Define at module level

# Note: These functions now need necessary info passed as arguments,
#       instead of accessing 'self'.

async def call_ollama(
    selected_ollama_model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    # loop: Optional[asyncio.AbstractEventLoop] = None, # loop is often implicit
    timeout: float = 180.0,  # Increased default timeout
    enable_thinking: bool = False # <<< NEW PARAMETER
) -> Tuple[Optional[str], Optional[str], Optional[str]]: # (content, thinking_trace, error_message)
    """
    Asynchronously calls the Ollama API chat endpoint.

    Args:
        selected_ollama_model: The name of the Ollama model to use.
        messages: A list of message dictionaries (e.g., [{"role": "user", "content": "..."}]).
        temperature: The temperature for generation.
        timeout: Timeout for the HTTP request in seconds.
        enable_thinking: If True, requests the model's thinking trace.

    Returns:
        A tuple containing:
            - The response content string (or None if error).
            - The thinking trace string (or None if not enabled/available or error).
            - An error message string (or None if successful).
    """
    api_url = f"{OLLAMA_API_BASE_URL}/api/chat"
    # Fallback for older Ollama versions or models not supporting /api/chat directly for generation
    # Might need a separate function or logic for /api/generate if chat doesn't work for all models
    # For now, assume /api/chat is primary.

    payload = {
        "model": selected_ollama_model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,  # Required for structured JSON response including 'thinking'
    }

    if enable_thinking:
        payload["think"] = True # Corrected parameter name as per Ollama documentation discussion
        logger_external_comms.debug(f"Ollama call to '{selected_ollama_model}' with 'think: true' enabled.")
    else:
        logger_external_comms.debug(f"Ollama call to '{selected_ollama_model}' with 'think: false'.")


    response_content: Optional[str] = None
    thinking_trace: Optional[str] = None # <<< NEW
    error_message: Optional[str] = None
    
    request_start_time = time.monotonic()

    try:
        # Ensure httpx is installed: pip install httpx
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger_external_comms.info(
                f"Calling Ollama API: model='{selected_ollama_model}', temp={temperature}, "
                f"timeout={timeout}s, thinking_enabled={enable_thinking}. "
                f"Prompt (user part snippet): '{messages[-1]['content'][:100]}...'"
            )
            response = await client.post(api_url, json=payload)
            
            response_duration = time.monotonic() - request_start_time
            logger_external_comms.info(
                f"Ollama API response for '{selected_ollama_model}' received in {response_duration:.2f}s. Status: {response.status_code}."
            )

            if response.status_code == 200:
                try:
                    response_data = response.json()
                    logger_external_comms.debug(f"Ollama raw JSON response: {str(response_data)[:500]}") # Log part of raw response
                    
                    # The chat endpoint typically returns the response within a "message" object
                    if "message" in response_data and isinstance(response_data["message"], dict):
                        message_obj = response_data["message"]
                        response_content = message_obj.get("content")
                        # The 'thinking' field might be at the top level of response_data, not inside message
                        thinking_trace = response_data.get("thinking") 
                        
                        if response_content is None:
                             logger_external_comms.warning(f"Ollama response for '{selected_ollama_model}' missing 'content' in 'message' object.")
                             error_message = "LLM response missing 'content' in 'message'."
                        if enable_thinking and thinking_trace is None:
                            logger_external_comms.warning(f"Thinking was enabled for '{selected_ollama_model}', but no 'thinking' field received in response object.")
                        elif thinking_trace:
                            logger_external_comms.info(f"Ollama thinking trace received for '{selected_ollama_model}' (len: {len(str(thinking_trace))}). Snippet: {str(thinking_trace)[:100]}...")

                    # Fallback for /api/generate or other structures (less common with `think:true`)
                    elif "response" in response_data and not response_content: 
                        response_content = response_data.get("response")
                        thinking_trace = response_data.get("thinking") # Check here too
                        logger_external_comms.warning(f"Ollama response for '{selected_ollama_model}' used 'response' field instead of 'message.content'. Thinking trace may be less reliable here.")
                        if enable_thinking and thinking_trace is None:
                            logger_external_comms.warning(f"Thinking was enabled for '{selected_ollama_model}', but no 'thinking' field received in top-level response.")
                    
                    elif not response_content: # If no content found by either key
                        error_message = "LLM response format unexpected or content field missing."
                        logger_external_comms.error(f"{error_message} Response Data: {str(response_data)[:500]}")
                        
                except json.JSONDecodeError as json_err:
                    error_message = f"Failed to decode Ollama JSON response: {json_err}. Response text: {response.text[:200]}"
                    logger_external_comms.error(error_message)
                except Exception as e_parse:
                    error_message = f"Error parsing Ollama response structure: {e_parse}. Response Data: {str(response_data)[:500] if 'response_data' in locals() else response.text[:500]}"
                    logger_external_comms.error(error_message, exc_info=True)
            else:
                error_message = f"Ollama API Error: Status {response.status_code} - {response.text[:200]}"
                logger_external_comms.error(error_message)

    except httpx.TimeoutException as timeout_err:
        error_message = f"Ollama API call timed out after {timeout}s for model {selected_ollama_model}: {timeout_err}"
        logger_external_comms.error(error_message)
    except httpx.RequestError as req_err:
        error_message = f"Ollama API request error for model {selected_ollama_model}: {req_err}"
        logger_external_comms.error(error_message)
    except Exception as e:
        error_message = f"Unexpected error calling Ollama API for model {selected_ollama_model}: {e}"
        logger_external_comms.error(error_message, exc_info=True)

    return response_content, thinking_trace, error_message


async def add_document_to_memory(
    memory_collection: Optional[Any], # Use Any for chromadb type safety
    loop: Optional[asyncio.AbstractEventLoop],
    text: str,
    metadata: Dict[str, Any],
    is_running_flag: Optional[threading.Event] # Allow optional flag
):
    """Adds a document to the ChromaDB memory if available and agent is running."""
    if not (CHROMADB_AVAILABLE and memory_collection and text):
        logger_external_comms.debug("Add document skipped: ChromaDB/collection/text missing.")
        return
    if not loop or not loop.is_running(): # Check flags and loop
         logger_external_comms.warning("Add document skipped: Loop not running or not provided.")
         return
    if is_running_flag and not is_running_flag.is_set():
        logger_external_comms.warning("Add document skipped: Agent not running.")
        return


    # Ensure metadata keys and values are suitable for ChromaDB
    # Filter out complex types, ensure basic types
    safe_metadata = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            safe_metadata[k] = v
        # Add specific handling for lists/tuples if needed by your Chroma version
        # elif isinstance(v, (list, tuple)):
        #    safe_metadata[k] = json.dumps(v) # Example: Serialize lists
        else:
             logger_external_comms.debug(f"Skipping metadata key '{k}' due to non-basic type: {type(v)}")


    if not safe_metadata.get("timestamp"): safe_metadata["timestamp"] = time.time()
    if not safe_metadata.get("type"): safe_metadata["type"] = "generic_doc"
    # Generate a more robust ID
    doc_id = f"{safe_metadata['type']}_{int(safe_metadata['timestamp'] * 1000)}_{random.randint(1000,9999)}"

    try:
        logger_external_comms.debug(f"Adding doc to memory: {doc_id} (Type: {safe_metadata['type']}, len {len(text)})")
        # Define sync function within scope, capturing collection
        add_func = lambda collection=memory_collection: collection.add(documents=[text], metadatas=[safe_metadata], ids=[doc_id])
        await loop.run_in_executor( None, add_func) # Use passed loop
        logger_external_comms.debug(f"Doc {doc_id} added to memory.")
    except RuntimeError as e:
         if "cannot schedule new futures after shutdown" in str(e): logger_external_comms.warning(f"Add document '{doc_id}' aborted: Executor shutdown.")
         else: logger_external_comms.exception(f"Failed add doc {doc_id} (RuntimeError)"); # Log locally
    except Exception as e:
        # Catch potential chromadb specific errors if the library defines them
        # Example: if isinstance(e, chromadb.errors.SomeError): ...
        logger_external_comms.exception(f"Failed add doc {doc_id}"); # Log locally


async def retrieve_from_memory(
    memory_collection: Optional[Any], # Use Any type
    loop: Optional[asyncio.AbstractEventLoop],
    query_text: str,
    is_running_flag: Optional[threading.Event],
    n_results: int = 3 # Use default from function signature if not in config
) -> List[str]:
    """Retrieves documents from ChromaDB memory if available and agent is running."""
    # Use default n_results if config not found
    n_results_actual = getattr(_agent_config_ext, 'MEMORY_RETRIEVAL_COUNT', n_results)

    if not (CHROMADB_AVAILABLE and memory_collection and query_text):
        logger_external_comms.debug("Retrieve skipped: ChromaDB/collection/query missing.")
        return []
    if not loop or not loop.is_running():
        logger_external_comms.warning("Retrieve skipped: Loop not running or not provided.")
        return []
    if is_running_flag and not is_running_flag.is_set():
        logger_external_comms.warning("Retrieve skipped: Agent not running.")
        return []

    try:
        logger_external_comms.debug(f"Querying memory (n={n_results_actual}): '{query_text[:50]}...'")
        # Define sync function within scope, capturing collection
        query_func = lambda collection=memory_collection: collection.query(
            query_texts=[query_text],
            n_results=n_results_actual,
            include=['documents', 'metadatas'] # Include metadata if needed later
            )
        results = await loop.run_in_executor( None, query_func) # Use passed loop

        retrieved_docs = []
        # Safely extract documents
        if results and isinstance(results.get('documents'), list) and results['documents']:
             doc_list = results['documents'][0] # Query returns list of lists
             if isinstance(doc_list, list):
                 retrieved_docs = [str(doc) if doc is not None else "[None]" for doc in doc_list]
        logger_external_comms.info(f"Retrieved {len(retrieved_docs)} docs from memory.");
        return retrieved_docs

    except RuntimeError as e:
        if "cannot schedule new futures after shutdown" in str(e): logger_external_comms.warning(f"Memory query aborted: Executor shutdown."); return []
        else: logger_external_comms.exception("ChromaDB query RuntimeError"); return [] # Log locally
    except Exception as e:
        # Catch potential chromadb specific errors
        logger_external_comms.exception("ChromaDB query failed"); return [] # Log locally
# --- END OF FILE external_comms.py ---