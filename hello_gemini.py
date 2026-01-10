#!/usr/bin/env python3
"""Quick hello world script to test Gemini via litellm."""

import os
import litellm
from litellm import completion

def main():
    # Get API key from environment - litellm expects GOOGLE_API_KEY for Gemini
    # Check both variable names for compatibility (user might have GEMINI_API_KEY)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
    
    # Ensure GOOGLE_API_KEY is set (litellm uses this for Google AI Studio)
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # Explicitly disable Vertex AI to force Google AI Studio routing
    # This prevents litellm from trying to use Vertex AI authentication
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        # Temporarily unset to force Google AI Studio instead of Vertex AI
        old_creds = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")
    
    # Make a simple hello world call
    # Use gemini/ prefix - with GOOGLE_API_KEY set, litellm will use Google AI Studio
    response = completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "user", "content": "Hello! Please respond with 'Hello, world!'"}
        ],
    )
    
    # Print the response
    print("Response from Gemini:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()
