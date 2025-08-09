#!/usr/bin/env python3
# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Quick-start example for using AWS Bedrock with langextract."""

import argparse
import os

import langextract as lx

# Clear the LRU cache to ensure fresh pattern resolution
from langextract.providers import registry
registry.resolve.cache_clear()


def run_extraction(
    model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    aws_region="us-east-1",
    aws_profile=None,
    temperature=0.0,
):
  """Run a simple extraction example using AWS Bedrock."""
  input_text = "Dr. Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity."

  prompt = "Extract the person's full name, their professional fields, and their primary research area."

  examples = [
      lx.data.ExampleData(
          text=(
              "Albert Einstein was a theoretical physicist who developed"
              " the theory of relativity."
          ),
          extractions=[
              lx.data.Extraction(
                  extraction_class="scientist_details",
                  # extraction_text includes full context with ellipsis for clarity
                  extraction_text="Albert Einstein was a theoretical physicist...",
                  attributes={
                      "name": "Albert Einstein",
                      "fields": ["theoretical physics"],
                      "research_area": "theory of relativity",
                  },
              )
          ],
      )
  ]

  result = lx.extract(
      text_or_documents=input_text,
      prompt_description=prompt,
      examples=examples,
      model_id=model_id,
      temperature=temperature,
      language_model_params={
          "aws_region": aws_region,
          "aws_profile": aws_profile,
      },
  )

  return result


def main():
  """Main function to run the quick-start example."""
  parser = argparse.ArgumentParser(description="Run AWS Bedrock extraction example")
  parser.add_argument(
      "--model-id",
      default=os.getenv("MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
      help="Bedrock model ID (default: us.anthropic.claude-3-5-sonnet-20241022-v2:0 or MODEL_ID env var)",
  )
  parser.add_argument(
      "--aws-region",
      default=os.getenv("AWS_REGION", "us-east-1"),
      help="AWS region (default: us-east-1 or AWS_REGION env var)",
  )
  parser.add_argument(
      "--aws-profile",
      default=os.getenv("AWS_PROFILE"),
      help="AWS profile (default: AWS_PROFILE env var)",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=float(os.getenv("TEMPERATURE", "0.0")),
      help="Model temperature (default: 0.0 or TEMPERATURE env var)",
  )
  parser.add_argument(
      "--list-models",
      action="store_true",
      help="Show available Bedrock model examples and exit",
  )

  args = parser.parse_args()

  if args.list_models:
    print("🤖 Supported Bedrock Model Families:")
    print("-" * 50)
    models = [
        ("Anthropic Claude", [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
        ]),
        ("Amazon Titan", [
            "amazon.titan-text-lite-v1",
            "amazon.titan-text-express-v1",
            "amazon.titan-embed-text-v2:0"
        ]),
        ("Meta Llama", [
            "meta.llama3-8b-instruct-v1:0",
            "meta.llama3-70b-instruct-v1:0",
        ]),
        ("Cohere Command", [
            "cohere.command-text-v14",
            "cohere.command-light-text-v14",
        ]),
        ("AI21 Labs", [
            "ai21.j2-mid-v1",
            "ai21.j2-ultra-v1",
        ]),
        ("Mistral", [
            "mistral.mistral-7b-instruct-v0:2",
            "mistral.mixtral-8x7b-instruct-v0:1",
        ]),
    ]
    
    for family, model_list in models:
      print(f"\n{family}:")
      for model in model_list:
        print(f"  • {model}")
    
    print(f"\nNote: Model availability varies by AWS region.")
    print(f"Check the AWS Bedrock console for models available in {args.aws_region}")
    return True

  print(f"🚀 Running AWS Bedrock quick-start example...")
  print(f"   Model: {args.model_id}")
  print(f"   Region: {args.aws_region}")
  if args.aws_profile:
    print(f"   Profile: {args.aws_profile}")
  print("-" * 50)

  try:
    result = run_extraction(
        model_id=args.model_id,
        aws_region=args.aws_region,
        aws_profile=args.aws_profile,
        temperature=args.temperature,
    )

    print(f"\n📄 Input: {result.text}")
    print(f"\n🎯 Extractions:")
    for extraction in result.extractions:
      print(f"  Class: {extraction.extraction_class}")
      print(f"  Text: {extraction.extraction_text}")
      print(f"  Attributes: {extraction.attributes}")

    print("\n✅ SUCCESS! AWS Bedrock is working with langextract")
    return True

  except ImportError as e:
    if "boto3" in str(e):
      print(f"\nImportError: {e}")
      print("Install AWS dependencies: pip install langextract[bedrock]")
    else:
      print(f"\nImportError: {e}")
    return False
  except Exception as e:
    error_type = type(e).__name__
    error_msg = str(e)
    
    print(f"\n{error_type}: {error_msg}")
    
    if "credentials" in error_msg.lower() or "access" in error_msg.lower():
      print("\n🔧 AWS Credentials Setup:")
      print("1. Set environment variables:")
      print("   export AWS_REGION=us-east-1")
      print("   export AWS_PROFILE=your-profile  # optional")
      print("\n2. Or configure AWS CLI:")
      print("   aws configure")
      print("\n3. Ensure your credentials have Bedrock permissions:")
      print("   bedrock:InvokeModel, bedrock:ListFoundationModels")
    
    elif "region" in error_msg.lower():
      print(f"\n🔧 Region Setup:")
      print(f"Bedrock is not available in all regions.")
      print(f"Try: --aws-region us-east-1 or --aws-region us-west-2")
    
    elif "model" in error_msg.lower() or "resource" in error_msg.lower():
      print(f"\n🔧 Model Availability:")
      print(f"The model '{args.model_id}' may not be available in '{args.aws_region}'")
      print(f"Run with --list-models to see examples")
      print(f"Check AWS Bedrock console for available models in your region")
    
    return False


if __name__ == "__main__":
  success = main()
  exit(0 if success else 1)
