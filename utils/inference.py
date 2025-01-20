import os
import openai
import anthropic
import aiohttp
import json
from together import AsyncTogether
from typing import Optional, Dict, Any
from openai import AsyncOpenAI
import asyncio
import requests




# Load environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
together_api_key = os.getenv('TOGETHER_API_KEY')
writer_api_key = os.getenv('WRITER_API_KEY')
hf_api_token = os.getenv('HF_API_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
#hf_client = InferenceClient(token=hf_api_token)  # Replace with your Hugging Face token

# Initialize clients only when API keys are available
async_openai_client = AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

# Initialize NVIDIA client if API key exists
nvidia_api_key = os.getenv('NVIDIA_API_KEY')

# Initialize DeepSeek client if API key exists
deepseek_client = AsyncOpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com"
) if deepseek_api_key else None

# Initialize DeepInfra client if API key exists
deepinfra_api_key = os.getenv('DEEPINFRA_API_KEY')
deepinfra_client = AsyncOpenAI(
    api_key=deepinfra_api_key,
    base_url="https://api.deepinfra.com/v1/openai"
) if deepinfra_api_key else None

async def generate_text(model: str, prompt: str, max_tokens: int = 8000, temperature: float = 0) -> str:
    print(f"Generating text with model: {model}")
    """
    Asynchronously generate text using various AI models.
    
    :param model: The name of the model to use (e.g., "gpt-3.5-turbo", "claude-2", "meta-llama/Llama-2-70b-chat-hf")
    :param prompt: The input prompt for text generation
    :param max_tokens: Maximum number of tokens to generate
    :param temperature: Controls randomness in generation (0.0 to 1.0)
    :return: Generated text as a string
    """
    
    # OpenAI models
    
   
    if model.startswith("gpt-") or model.startswith("o1"):
        try:
            response = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"[ERROR] OpenAI API error for {model}: {str(e)}")
            raise
    
    elif model.startswith("ft:gpt") or model.startswith("o1"):
        try:
            response = await async_openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"[ERROR] Fine-tuned OpenAI API error for {model}: {str(e)}")
            raise
    
    # Anthropic (Claude) models
    elif model.startswith("claude-"):
        async def run_anthropic():
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            if model.startswith("claude-3"):
                response = client.messages.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.content[0].text.strip()
            else:
                response = client.completions.create(
                    model=model,
                    prompt=f"Human: {prompt}\n\nAssistant:",
                    max_tokens_to_sample=max_tokens,
                    temperature=temperature
                )
                return response.completion.strip()
        
        return await run_anthropic()
    
    # Together AI models

   
    elif model.startswith("meta-llama/") or model.startswith("Qwen/") or model.startswith("Meta-Llama"):
        response = await deepinfra_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    
    
    
    # NVIDIA models
    elif model.startswith("nvidia"):
        nvidia_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key
        )
        
        response = await nvidia_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        
        return full_response.strip()

    # Add DeepSeek model support after the NVIDIA section
    elif model.startswith("deepseek-"):
        response = await deepseek_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    # Hugging Face models (fallback for any other model)
    else:
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {hf_api_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
        async def query_huggingface():
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0 and 'generated_text' in result[0]:
                            return result[0]['generated_text'].strip()
                        else:
                            return str(result)
                    else:
                        raise Exception(f"Error with Hugging Face Inference API for model {model}: {await response.text()}")
        
        return await query_huggingface()
