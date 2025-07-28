#!/usr/bin/env python3
"""
Simple test script for the PDF to Text API
"""

import requests
import sys
import json

def test_health_check(base_url="http://localhost:5000"):
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{base_url}/")
        print(f"Health Check Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_pdf_conversion(pdf_path, base_url="http://localhost:5000"):
    """Test PDF conversion endpoint"""
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/convert", files=files)
        
        print(f"Conversion Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Conversion successful!")
                print(f"Merged text length: {len(result.get('merged_text', ''))}")
                print(f"Number of pages: {len(result.get('page_details', []))}")
                
                # Show first 500 characters of merged text
                merged_text = result.get('merged_text', '')
                if merged_text:
                    print(f"\nFirst 500 characters of merged text:")
                    print("-" * 50)
                    print(merged_text[:500])
                    if len(merged_text) > 500:
                        print("...")
                    print("-" * 50)
                
                # Show image count per page
                for page in result.get('page_details', []):
                    image_count = len(page.get('images', []))
                    if image_count > 0:
                        print(f"Page {page['page_number']}: {image_count} image(s) processed")
                
                return True
            else:
                print(f"❌ Conversion failed: {result.get('error')}")
                return False
        else:
            print(f"❌ HTTP Error: {response.text}")
            return False
            
    except FileNotFoundError:
        print(f"❌ File not found: {pdf_path}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing PDF to Text API")
    print("=" * 40)
    
    # Test health check
    print("\n1. Testing health check...")
    if not test_health_check(base_url):
        print("❌ Health check failed. Is the server running?")
        sys.exit(1)
    
    # Test PDF conversion if file provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\n2. Testing PDF conversion with: {pdf_path}")
        if test_pdf_conversion(pdf_path, base_url):
            print("✅ All tests passed!")
        else:
            print("❌ PDF conversion test failed!")
            sys.exit(1)
    else:
        print("\n2. Skipping PDF conversion test (no file provided)")
        print("   Usage: python test_api.py <path_to_pdf_file>")
    
    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    main()