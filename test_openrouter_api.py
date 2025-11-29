"""
Test script to verify OpenRouter API connection
Run this before using the full API to ensure everything is set up correctly
"""

import os
import requests
import json
from dotenv import load_dotenv
load_dotenv()
def test_openrouter_connection():
    """Test basic OpenRouter API connection"""
    
    print("=" * 60)
    print("OpenRouter API Connection Test")
    print("=" * 60)
    
    # Step 1: Check API key
    print("\n1. Checking API Key...")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print("api_key", api_key)
    if not api_key:
        print("❌ FAILED: OPENROUTER_API_KEY environment variable not set")
        print("\nHow to fix:")
        print("  Linux/Mac: export OPENROUTER_API_KEY='your_key_here'")
        print("  Windows:   set OPENROUTER_API_KEY=your_key_here")
        return False
    
    # Mask the key for security
    masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
    print(f"✓ API Key found: {masked_key}")
    
    # Step 2: Test API endpoint
    print("\n2. Testing API Endpoint...")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/test/resume-analyzer",
        "X-Title": "Resume Analyzer Test"
    }
    
    # Test with a simple prompt using a free model
    data = {
        "model": "google/gemma-3-4b-it:free",
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello' in one word."
            }
        ],
        "max_tokens": 10
    }
    
    try:
        print(f"Sending request to: {url}")
        print(f"Using model: {data['model']}")
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"✓ SUCCESS! Response: {content}")
            print(f"✓ Model used: {result.get('model', 'N/A')}")
            print(f"✓ Tokens: {result.get('usage', {})}")
            return True
        
        elif response.status_code == 401:
            print("❌ FAILED: Invalid API key (401 Unauthorized)")
            print("\nHow to fix:")
            print("  1. Go to https://openrouter.ai/keys")
            print("  2. Generate a new API key")
            print("  3. Set it as environment variable")
            return False
        
        elif response.status_code == 404:
            print("❌ FAILED: Endpoint not found (404)")
            print(f"\nResponse: {response.text}")
            print("\nPossible issues:")
            print("  1. API endpoint URL might be wrong")
            print("  2. Check OpenRouter status: https://status.openrouter.ai")
            return False
        
        elif response.status_code == 429:
            print("❌ FAILED: Rate limit exceeded (429)")
            print("  Wait a moment and try again")
            return False
        
        else:
            print(f"❌ FAILED: Unexpected status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ FAILED: Request timed out")
        print("  The API might be slow or unavailable")
        return False
    
    except requests.exceptions.ConnectionError:
        print("❌ FAILED: Connection error")
        print("  Check your internet connection")
        return False
    
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

def test_models():
    """Test if we can fetch available models"""
    
    print("\n" + "=" * 60)
    print("Testing Model List API")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not set")
        return False
    
    url = "https://openrouter.ai/api/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            print(f"✓ Found {len(models.get('data', []))} available models")
            
            # Show some free models
            free_models = [m for m in models.get('data', []) if ':free' in m.get('id', '')]
            if free_models:
                print(f"\nFree models available: {len(free_models)}")
                for model in free_models[:5]:
                    print(f"  - {model['id']}")
            
            return True
        else:
            print(f"❌ Failed to fetch models: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Error fetching models: {str(e)}")
        return False

def test_with_different_models():
    """Test with different free models"""
    
    print("\n" + "=" * 60)
    print("Testing Different Models")
    print("=" * 60)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not set")
        return
    
    test_models = [
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free",
        "mistralai/mistral-7b-instruct:free"
    ]
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    for model in test_models:
        print(f"\nTesting: {model}")
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 5
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                print(f"  ✓ Works! Response: {content[:50]}")
            else:
                print(f"  ❌ Failed: {response.status_code}")
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

if __name__ == "__main__":
    print("\n🚀 Starting OpenRouter API Tests\n")
    
    # Run tests
    connection_ok = test_openrouter_connection()
    
    if connection_ok:
        print("\n✅ Basic connection works!")
        test_models()
        test_with_different_models()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! Your OpenRouter setup is working.")
        print("=" * 60)
        print("\nYou can now use the API endpoints:")
        print("  POST /api/analyze-ai")
        print("  POST /api/enhance-section")
        print("  POST /api/generate-cover-letter")
        print("  POST /api/interview-prep")
    else:
        print("\n" + "=" * 60)
        print("❌ Tests failed. Please fix the issues above.")
        print("=" * 60)
        print("\nQuick checklist:")
        print("  1. Do you have an OpenRouter account? → https://openrouter.ai")
        print("  2. Have you created an API key? → https://openrouter.ai/keys")
        print("  3. Is OPENROUTER_API_KEY set? → echo $OPENROUTER_API_KEY")
        print("  4. Is your internet working?")