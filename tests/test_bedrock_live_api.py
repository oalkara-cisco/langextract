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

"""Live API tests for AWS Bedrock provider.

These tests require AWS credentials and region to be configured.
They are skipped if AWS_REGION is not set.
"""

import os
import time
import unittest
from functools import wraps

from dotenv import load_dotenv
import pytest

import langextract as lx

load_dotenv()

DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_REGION = os.environ.get("AWS_REGION")
AWS_PROFILE = os.environ.get("AWS_PROFILE")

skip_if_no_bedrock = pytest.mark.skipif(
    not AWS_REGION,
    reason="AWS Bedrock not available (set AWS_REGION and configure AWS credentials)",
)

live_api = pytest.mark.live_api

INITIAL_RETRY_DELAY = 1.0
MAX_RETRY_DELAY = 8.0


def retry_on_transient_errors(max_retries=3, backoff_factor=2.0):
  """Decorator to retry tests on transient API errors with exponential backoff."""

  def decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      last_exception = None
      delay = INITIAL_RETRY_DELAY
      
      for attempt in range(max_retries + 1):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          last_exception = e
          error_str = str(e)
          
          # Check for transient errors
          transient_patterns = [
              "throttled",
              "rate limit",
              "quota",
              "temporarily",
              "timeout",
              "connection",
              "429",  # Too Many Requests
              "503",  # Service Unavailable
              "502",  # Bad Gateway
          ]
          
          is_transient = any(
              pattern in error_str.lower() for pattern in transient_patterns
          )
          
          if is_transient and attempt < max_retries:
            print(
                f"\nTransient error on attempt {attempt + 1}/{max_retries + 1}: {error_str}"
            )
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * backoff_factor, MAX_RETRY_DELAY)
          else:
            raise
      
      # Should never reach here, but just in case
      if last_exception:
        raise last_exception

    return wrapper

  return decorator


class TestLiveAPIBedrock(unittest.TestCase):
  """Live API tests for AWS Bedrock provider."""

  @live_api
  @skip_if_no_bedrock
  def test_bedrock_basic_extraction(self):
    """Test basic text extraction using Bedrock."""
    text = "The patient was prescribed Aspirin 100mg daily for pain management."

    result = lx.extract(
        text,
        ["medication", "dosage", "indication"],
        model_id=DEFAULT_BEDROCK_MODEL,
        aws_region=AWS_REGION,
        aws_profile=AWS_PROFILE,
        temperature=0.0,
        annotation_style="markdown",
    )

    self.assertIsNotNone(result)
    self.assertGreater(len(result.extractions), 0)

    # Verify extraction types
    extraction_types = {e.extraction_class for e in result.extractions}
    self.assertIn("medication", extraction_types)
    self.assertIn("dosage", extraction_types)

  @live_api
  @skip_if_no_bedrock
  @retry_on_transient_errors(max_retries=3)
  def test_bedrock_structured_output(self):
    """Test structured output with schemas using Bedrock."""
    
    medication_schema = lx.Schema(
        name="Medication",
        description="A medication prescription",
        attributes=[
            lx.Attribute(
                name="drug_name",
                description="Name of the medication",
                value_type="string",
                required=True,
            ),
            lx.Attribute(
                name="dosage",
                description="Dosage amount and frequency",
                value_type="string",
                required=True,
            ),
            lx.Attribute(
                name="indication",
                description="Medical condition being treated",
                value_type="string",
                required=False,
            ),
        ],
    )

    text = "The doctor prescribed Metformin 500mg twice daily for type 2 diabetes management."

    result = lx.extract(
        text,
        [medication_schema],
        model_id=DEFAULT_BEDROCK_MODEL,
        aws_region=AWS_REGION,
        aws_profile=AWS_PROFILE,
        temperature=0.0,
    )

    self.assertIsNotNone(result)
    self.assertGreater(len(result.extractions), 0)

    # Check structured data
    medication = result.extractions[0]
    self.assertEqual(medication.extraction_class, "Medication")
    self.assertIn("drug_name", medication.attributes)
    self.assertIn("dosage", medication.attributes)
    self.assertIsInstance(medication.attributes["drug_name"], str)
    self.assertIsInstance(medication.attributes["dosage"], str)

  @live_api
  @skip_if_no_bedrock
  def test_bedrock_multiple_providers(self):
    """Test different Bedrock model providers."""
    text = "The quick brown fox jumps over the lazy dog."
    
    # Test with different model IDs (only if available)
    test_models = [
        "anthropic.claude-3-haiku-20240307-v1:0",  # Faster, cheaper Claude model
        "amazon.titan-text-lite-v1",  # Amazon's model if available
    ]
    
    for model_id in test_models:
      try:
        result = lx.extract(
            text,
            ["animal", "action"],
            model_id=model_id,
            aws_region=AWS_REGION,
            aws_profile=AWS_PROFILE,
            temperature=0.0,
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(len(result.extractions), 0)
        
        # Basic validation
        extraction_types = {e.extraction_class for e in result.extractions}
        self.assertTrue(
            "animal" in extraction_types or "action" in extraction_types,
            f"Model {model_id} failed to extract expected entities"
        )
      except Exception as e:
        # Some models might not be available in all regions
        if "ResourceNotFoundException" in str(e) or "ValidationException" in str(e):
          self.skipTest(f"Model {model_id} not available in region {AWS_REGION}")
        else:
          raise

  @live_api
  @skip_if_no_bedrock
  def test_bedrock_batch_processing(self):
    """Test batch processing with Bedrock."""
    texts = [
        "Patient A was prescribed Lisinopril 10mg daily.",
        "Patient B takes Metformin 1000mg twice daily.",
        "Patient C uses Insulin glargine 20 units at bedtime.",
    ]

    results = lx.extract_batch(
        texts,
        ["medication", "dosage"],
        model_id=DEFAULT_BEDROCK_MODEL,
        aws_region=AWS_REGION,
        aws_profile=AWS_PROFILE,
        temperature=0.0,
        max_workers=2,  # Test parallel processing
    )

    self.assertEqual(len(results), 3)
    
    for i, result in enumerate(results):
      self.assertIsNotNone(result, f"Result {i} is None")
      self.assertGreater(
          len(result.extractions), 0,
          f"No extractions found in text {i}"
      )
      
      # Each text should have at least one medication
      extraction_types = {e.extraction_class for e in result.extractions}
      self.assertIn("medication", extraction_types)

  @live_api
  @skip_if_no_bedrock
  def test_bedrock_output_formats(self):
    """Test different output formats with Bedrock."""
    text = "Prescribe Aspirin 81mg once daily for cardiovascular protection."
    
    # Test JSON format
    result_json = lx.extract(
        text,
        ["medication", "dosage", "indication"],
        model_id=DEFAULT_BEDROCK_MODEL,
        aws_region=AWS_REGION,
        aws_profile=AWS_PROFILE,
        format_type=lx.FormatType.JSON,
        temperature=0.0,
    )
    
    self.assertIsNotNone(result_json)
    self.assertGreater(len(result_json.extractions), 0)
    
    # Test YAML format
    result_yaml = lx.extract(
        text,
        ["medication", "dosage", "indication"],
        model_id=DEFAULT_BEDROCK_MODEL,
        aws_region=AWS_REGION,
        aws_profile=AWS_PROFILE,
        format_type=lx.FormatType.YAML,
        temperature=0.0,
    )
    
    self.assertIsNotNone(result_yaml)
    self.assertGreater(len(result_yaml.extractions), 0)
    
    # Both formats should extract similar entities
    json_types = {e.extraction_class for e in result_json.extractions}
    yaml_types = {e.extraction_class for e in result_yaml.extractions}
    
    # Should have at least medication in both
    self.assertIn("medication", json_types)
    self.assertIn("medication", yaml_types)

  @live_api
  @skip_if_no_bedrock
  @retry_on_transient_errors(max_retries=3)
  def test_bedrock_error_handling(self):
    """Test error handling for invalid model IDs."""
    with self.assertRaises(Exception) as context:
      lx.extract(
          "Test text",
          ["entity"],
          model_id="invalid.model-id",
          aws_region=AWS_REGION,
          aws_profile=AWS_PROFILE,
      )
    
    # Should get an error about unsupported model
    self.assertIn("Unsupported model provider", str(context.exception))


if __name__ == "__main__":
  unittest.main()
