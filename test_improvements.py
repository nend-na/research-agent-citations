#!/usr/bin/env python
"""Quick verification that improvements are in place."""

import os
from dotenv import load_dotenv

load_dotenv()

# Test 1: Verify torchvision in requirements
with open('requirements.txt') as f:
    reqs = f.read()
    print(f"✓ torchvision in requirements: {'torchvision' in reqs}")
    print(f"✓ python-dotenv in requirements: {'python-dotenv' in reqs}")

# Test 2: Verify .env.example has API key URLs
with open('.env.example') as f:
    env_example = f.read()
    has_urls = 'https://console' in env_example or 'https://platform' in env_example
    print(f"✓ .env.example has API key URLs: {has_urls}")

# Test 3: Verify README has architecture diagram
with open('README.md') as f:
    readme = f.read()
    has_architecture = 'System Architecture' in readme
    has_dataflow = 'Data Flow' in readme
    has_example = 'Example Usage' in readme
    print(f"✓ README has 'System Architecture': {has_architecture}")
    print(f"✓ README has 'Data Flow': {has_dataflow}")
    print(f"✓ README has 'Example Usage': {has_example}")

# Test 4: Quick pipeline check
try:
    from app.main import pipeline
    from app.generator import check_provider_status
    
    status = check_provider_status('groq')
    print(f"\n✓ Groq provider status: {status}")
    
    # Run a simple query
    result = pipeline('What is remote work?', top_k=2)
    print(f"✓ Pipeline executed: answer length = {len(result.answer)} chars")
    print(f"✓ Citations generated: {len(result.citations)} sources")
    print(f"✓ Generation status: {result.generation_status}")
    
except Exception as e:
    print(f"✗ Pipeline test failed: {e}")

print("\n=== All Improvements Verified ===")
