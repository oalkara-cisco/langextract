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

"""Tests for AWS Bedrock provider."""

import json
import os
from unittest import mock

from absl.testing import absltest

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract.providers import bedrock


class BedrockLanguageModelTest(absltest.TestCase):
  """Tests for BedrockLanguageModel."""

  def setUp(self):
    super().setUp()
    # Mock the boto3 import
    self.mock_boto3 = mock.MagicMock()
    self.mock_client = mock.MagicMock()
    self.mock_session = mock.MagicMock()
    self.mock_session.client.return_value = self.mock_client
    self.mock_boto3.Session.return_value = self.mock_session

    # Patch boto3 import
    modules = {
        'boto3': self.mock_boto3,
    }
    self.module_patcher = mock.patch.dict('sys.modules', modules)
    self.module_patcher.start()

  def tearDown(self):
    super().tearDown()
    self.module_patcher.stop()

  def test_init_without_region(self):
    """Test initialization fails without AWS region."""
    with mock.patch.dict(os.environ, {}, clear=True):
      with self.assertRaisesRegex(
          exceptions.InferenceConfigError,
          'AWS region not provided'
      ):
        bedrock.BedrockLanguageModel()

  def test_init_with_explicit_params(self):
    """Test initialization with explicit parameters."""
    model = bedrock.BedrockLanguageModel(
        model_id='anthropic.claude-3-sonnet-20240229-v1:0',
        aws_profile='test-profile',
        aws_region='us-east-1',
        temperature=0.5,
        max_workers=5
    )

    self.assertEqual(model.model_id, 'anthropic.claude-3-sonnet-20240229-v1:0')
    self.assertEqual(model.aws_profile, 'test-profile')
    self.assertEqual(model.aws_region, 'us-east-1')
    self.assertEqual(model.temperature, 0.5)
    self.assertEqual(model.max_workers, 5)
    self.assertEqual(model.format_type, data.FormatType.JSON)

    # Verify boto3 session was created with correct params
    self.mock_boto3.Session.assert_called_once_with(
        profile_name='test-profile',
        region_name='us-east-1'
    )
    self.mock_session.client.assert_called_once_with('bedrock-runtime')

  def test_init_with_env_vars(self):
    """Test initialization with environment variables."""
    with mock.patch.dict(os.environ, {
        'AWS_PROFILE': 'env-profile',
        'AWS_REGION': 'us-west-2'
    }):
      model = bedrock.BedrockLanguageModel()

      self.assertEqual(model.aws_profile, 'env-profile')
      self.assertEqual(model.aws_region, 'us-west-2')

  def test_anthropic_request_body(self):
    """Test request body preparation for Anthropic Claude models."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0'
      )

      prompt = 'Test prompt'
      config = {
          'temperature': 0.7,
          'max_output_tokens': 2048,
          'top_p': 0.9,
          'top_k': 40
      }

      body = model._prepare_request_body(prompt, config)

      self.assertEqual(body['anthropic_version'], 'bedrock-2023-05-31')
      self.assertEqual(body['max_tokens'], 2048)
      self.assertEqual(body['temperature'], 0.7)
      self.assertEqual(body['top_p'], 0.9)
      self.assertEqual(body['top_k'], 40)
      self.assertEqual(len(body['messages']), 1)
      self.assertEqual(body['messages'][0]['role'], 'user')
      self.assertIn('Test prompt', body['messages'][0]['content'][0]['text'])
      self.assertIn('JSON format', body['messages'][0]['content'][0]['text'])

  def test_amazon_titan_request_body(self):
    """Test request body preparation for Amazon Titan models."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='amazon.titan-text-express-v1',
          format_type=data.FormatType.YAML
      )

      prompt = 'Test prompt'
      config = {
          'temperature': 0.5,
          'max_output_tokens': 1024,
          'top_p': 0.95
      }

      body = model._prepare_request_body(prompt, config)

      self.assertIn('Test prompt', body['inputText'])
      self.assertIn('YAML format', body['inputText'])
      self.assertEqual(body['textGenerationConfig']['maxTokenCount'], 1024)
      self.assertEqual(body['textGenerationConfig']['temperature'], 0.5)
      self.assertEqual(body['textGenerationConfig']['topP'], 0.95)

  def test_meta_llama_request_body(self):
    """Test request body preparation for Meta Llama models."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='meta.llama3-70b-instruct-v1:0'
      )

      prompt = 'Test prompt'
      config = {
          'temperature': 0.8,
          'max_output_tokens': 2048,
          'top_p': 0.9
      }

      body = model._prepare_request_body(prompt, config)

      self.assertIn('Test prompt', body['prompt'])
      self.assertEqual(body['max_gen_len'], 2048)
      self.assertEqual(body['temperature'], 0.8)
      self.assertEqual(body['top_p'], 0.9)

  def test_single_prompt_inference(self):
    """Test inference with a single prompt."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0'
      )

      # Mock the Bedrock response
      mock_response = {
          'body': mock.MagicMock()
      }
      mock_response['body'].read.return_value = json.dumps({
          'content': [{'text': 'Generated response'}]
      }).encode('utf-8')
      self.mock_client.invoke_model.return_value = mock_response

      # Run inference
      results = list(model.infer(['Test prompt']))

      # Verify results
      self.assertEqual(len(results), 1)
      self.assertEqual(len(results[0]), 1)
      self.assertEqual(results[0][0].output, 'Generated response')
      self.assertEqual(results[0][0].score, 1.0)

      # Verify API call
      self.mock_client.invoke_model.assert_called_once()
      call_args = self.mock_client.invoke_model.call_args
      self.assertEqual(call_args.kwargs['modelId'], 'anthropic.claude-3-sonnet-20240229-v1:0')
      self.assertEqual(call_args.kwargs['contentType'], 'application/json')
      self.assertEqual(call_args.kwargs['accept'], 'application/json')

  def test_batch_inference_parallel(self):
    """Test parallel batch inference."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0',
          max_workers=2
      )

      # Mock different responses for each prompt
      def mock_invoke_model(**kwargs):
        body = json.loads(kwargs['body'])
        prompt_text = body['messages'][0]['content'][0]['text']
        response_text = f"Response to: {prompt_text[:10]}..."
        
        mock_response = {'body': mock.MagicMock()}
        mock_response['body'].read.return_value = json.dumps({
            'content': [{'text': response_text}]
        }).encode('utf-8')
        return mock_response

      self.mock_client.invoke_model.side_effect = mock_invoke_model

      # Run batch inference
      prompts = ['Prompt 1', 'Prompt 2', 'Prompt 3']
      results = list(model.infer(prompts))

      # Verify results
      self.assertEqual(len(results), 3)
      for i, result in enumerate(results):
        self.assertEqual(len(result), 1)
        self.assertIn(f'Prompt {i+1}'[:10], result[0].output)

      # Verify parallel execution
      self.assertEqual(self.mock_client.invoke_model.call_count, 3)

  def test_fence_output(self):
    """Test output fencing for JSON and YAML."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      # Test JSON fencing
      model_json = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0',
          fence_output=True,
          format_type=data.FormatType.JSON
      )

      mock_response = {'body': mock.MagicMock()}
      mock_response['body'].read.return_value = json.dumps({
          'content': [{'text': '{"key": "value"}'}]
      }).encode('utf-8')
      self.mock_client.invoke_model.return_value = mock_response

      results = list(model_json.infer(['Test']))
      self.assertEqual(results[0][0].output, '```json\n{"key": "value"}\n```')

      # Test YAML fencing
      model_yaml = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0',
          fence_output=True,
          format_type=data.FormatType.YAML
      )

      mock_response['body'].read.return_value = json.dumps({
          'content': [{'text': 'key: value'}]
      }).encode('utf-8')
      self.mock_client.invoke_model.return_value = mock_response

      results = list(model_yaml.infer(['Test']))
      self.assertEqual(results[0][0].output, '```yaml\nkey: value\n```')

  def test_unsupported_model_provider(self):
    """Test error handling for unsupported model providers."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='unsupported.model-v1'
      )

      with self.assertRaisesRegex(
          exceptions.InferenceConfigError,
          'Unsupported model provider: unsupported'
      ):
        model._prepare_request_body('Test prompt', {})

  def test_api_error_handling(self):
    """Test handling of API errors."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      model = bedrock.BedrockLanguageModel(
          model_id='anthropic.claude-3-sonnet-20240229-v1:0'
      )

      # Mock an API error
      self.mock_client.invoke_model.side_effect = Exception('API Error')

      with self.assertRaisesRegex(
          exceptions.InferenceRuntimeError,
          'Bedrock API error: API Error'
      ):
        list(model.infer(['Test prompt']))

  def test_extract_response_for_all_providers(self):
    """Test response extraction for all supported providers."""
    with mock.patch.dict(os.environ, {'AWS_REGION': 'us-east-1'}):
      test_cases = [
          # (model_id, response_body, expected_text)
          (
              'anthropic.claude-3-sonnet-20240229-v1:0',
              {'content': [{'text': 'Claude response'}]},
              'Claude response'
          ),
          (
              'amazon.titan-text-express-v1',
              {'results': [{'outputText': 'Titan response'}]},
              'Titan response'
          ),
          (
              'meta.llama3-70b-instruct-v1:0',
              {'generation': 'Llama response'},
              'Llama response'
          ),
          (
              'cohere.command-text-v14',
              {'generations': [{'text': 'Cohere response'}]},
              'Cohere response'
          ),
          (
              'ai21.j2-ultra-v1',
              {'completions': [{'data': {'text': 'AI21 response'}}]},
              'AI21 response'
          ),
          (
              'mistral.mistral-7b-instruct-v0:2',
              {'outputs': [{'text': 'Mistral response'}]},
              'Mistral response'
          ),
      ]

      for model_id, response_body, expected_text in test_cases:
        model = bedrock.BedrockLanguageModel(model_id=model_id)
        extracted_text = model._extract_response_text(response_body)
        self.assertEqual(
            extracted_text,
            expected_text,
            f'Failed for model {model_id}'
        )


if __name__ == '__main__':
  absltest.main()
