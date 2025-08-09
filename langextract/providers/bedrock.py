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

"""AWS Bedrock provider for LangExtract."""
# pylint: disable=cyclic-import,duplicate-code

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
from typing import Any, Iterator, Sequence

from langextract import data
from langextract import exceptions
from langextract import inference
from langextract import schema
from langextract.providers import registry


@registry.register(
    r'^anthropic\.claude',  # anthropic.claude-3-sonnet, anthropic.claude-3-haiku, etc.
    r'^amazon\.titan',      # amazon.titan-text-lite, amazon.titan-text-express, etc.
    r'^meta\.llama',        # meta.llama3-8b, meta.llama3-70b, etc.
    r'^cohere\.command',    # cohere.command-text, cohere.command-light, etc.
    r'^ai21\.j2',           # ai21.j2-mid, ai21.j2-ultra, etc.
    r'^mistral\.',          # mistral.mistral-7b, mistral.mixtral-8x7b, etc.
    priority=10,
)
@dataclasses.dataclass(init=False)
class BedrockLanguageModel(inference.BaseLanguageModel):
  """Language model inference using AWS Bedrock API with structured output."""

  model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0'
  aws_profile: str | None = None
  aws_region: str | None = None
  format_type: data.FormatType = data.FormatType.JSON
  temperature: float = 0.0
  max_workers: int = 10
  fence_output: bool = False
  _client: Any = dataclasses.field(default=None, repr=False, compare=False)
  _extra_kwargs: dict[str, Any] = dataclasses.field(
      default_factory=dict, repr=False, compare=False
  )

  def __init__(
      self,
      model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0',
      aws_profile: str | None = None,
      aws_region: str | None = None,
      format_type: data.FormatType = data.FormatType.JSON,
      temperature: float = 0.0,
      max_workers: int = 10,
      fence_output: bool = False,
      **kwargs,
  ) -> None:
    """Initialize the Bedrock language model.

    Args:
      model_id: The Bedrock model ID to use.
      aws_profile: AWS profile name (defaults to AWS_PROFILE env var).
      aws_region: AWS region (defaults to AWS_REGION env var).
      format_type: Output format (JSON or YAML).
      temperature: Sampling temperature.
      max_workers: Maximum number of parallel API calls.
      fence_output: Whether to wrap output in markdown fences.
      **kwargs: Ignored extra parameters so callers can pass a superset of
        arguments shared across back-ends without raising ``TypeError``.
    """
    try:
      # pylint: disable=import-outside-toplevel
      import boto3
    except ImportError as e:
      raise exceptions.InferenceConfigError(
          'AWS Bedrock provider requires boto3 package. '
          'Install with: pip install langextract[bedrock]'
      ) from e

    self.model_id = model_id
    self.aws_profile = aws_profile or os.getenv('AWS_PROFILE')
    self.aws_region = aws_region or os.getenv('AWS_REGION')
    self.format_type = format_type
    self.temperature = temperature
    self.max_workers = max_workers
    self.fence_output = fence_output
    self._extra_kwargs = kwargs or {}

    if not self.aws_region:
      raise exceptions.InferenceConfigError(
          'AWS region not provided. Set AWS_REGION environment variable or '
          'pass aws_region parameter.'
      )

    # Initialize the Bedrock client
    session_kwargs = {}
    if self.aws_profile:
      session_kwargs['profile_name'] = self.aws_profile
    if self.aws_region:
      session_kwargs['region_name'] = self.aws_region

    session = boto3.Session(**session_kwargs)
    self._client = session.client('bedrock-runtime')

    super().__init__(
        constraint=schema.Constraint(constraint_type=schema.ConstraintType.NONE)
    )

  def _prepare_request_body(self, prompt: str, config: dict) -> dict:
    """Prepare the request body based on the model provider."""
    model_provider = self.model_id.split('.')[0].lower()

    # Add format instructions to the prompt
    format_instruction = ''
    if self.format_type == data.FormatType.JSON:
      format_instruction = '\n\nPlease respond with valid JSON format.'
    elif self.format_type == data.FormatType.YAML:
      format_instruction = '\n\nPlease respond with valid YAML format.'

    formatted_prompt = prompt + format_instruction

    if model_provider == 'anthropic':
      # Claude models
      body = {
          'anthropic_version': 'bedrock-2023-05-31',
          'max_tokens': config.get('max_output_tokens', 4096),
          'temperature': config.get('temperature', self.temperature),
          'messages': [{
              'role': 'user',
              'content': [{
                  'type': 'text',
                  'text': formatted_prompt
              }]
          }]
      }
      if 'top_p' in config:
        body['top_p'] = config['top_p']
      if 'top_k' in config:
        body['top_k'] = config['top_k']

    elif model_provider == 'amazon':
      # Titan models
      body = {
          'inputText': formatted_prompt,
          'textGenerationConfig': {
              'maxTokenCount': config.get('max_output_tokens', 4096),
              'temperature': config.get('temperature', self.temperature),
          }
      }
      if 'top_p' in config:
        body['textGenerationConfig']['topP'] = config['top_p']

    elif model_provider == 'meta':
      # Llama models
      body = {
          'prompt': formatted_prompt,
          'max_gen_len': config.get('max_output_tokens', 2048),
          'temperature': config.get('temperature', self.temperature),
      }
      if 'top_p' in config:
        body['top_p'] = config['top_p']

    elif model_provider == 'cohere':
      # Cohere models
      body = {
          'prompt': formatted_prompt,
          'max_tokens': config.get('max_output_tokens', 4096),
          'temperature': config.get('temperature', self.temperature),
      }
      if 'top_p' in config:
        body['p'] = config['top_p']
      if 'top_k' in config:
        body['k'] = config['top_k']

    elif model_provider == 'ai21':
      # AI21 models
      body = {
          'prompt': formatted_prompt,
          'maxTokens': config.get('max_output_tokens', 8192),
          'temperature': config.get('temperature', self.temperature),
      }
      if 'top_p' in config:
        body['topP'] = config['top_p']

    elif model_provider == 'mistral':
      # Mistral models
      body = {
          'prompt': f'<s>[INST] {formatted_prompt} [/INST]',
          'max_tokens': config.get('max_output_tokens', 8192),
          'temperature': config.get('temperature', self.temperature),
      }
      if 'top_p' in config:
        body['top_p'] = config['top_p']
      if 'top_k' in config:
        body['top_k'] = config['top_k']

    else:
      raise exceptions.InferenceConfigError(
          f'Unsupported model provider: {model_provider}'
      )

    return body

  def _extract_response_text(self, response_body: dict) -> str:
    """Extract text from the response based on the model provider."""
    model_provider = self.model_id.split('.')[0].lower()

    if model_provider == 'anthropic':
      # Claude models
      content = response_body.get('content', [])
      return ''.join(item.get('text', '') for item in content)

    elif model_provider == 'amazon':
      # Titan models
      results = response_body.get('results', [])
      if results:
        return results[0].get('outputText', '')
      return ''

    elif model_provider == 'meta':
      # Llama models
      return response_body.get('generation', '')

    elif model_provider == 'cohere':
      # Cohere models
      generations = response_body.get('generations', [])
      if generations:
        return generations[0].get('text', '')
      return ''

    elif model_provider == 'ai21':
      # AI21 models
      completions = response_body.get('completions', [])
      if completions:
        return completions[0].get('data', {}).get('text', '')
      return ''

    elif model_provider == 'mistral':
      # Mistral models
      outputs = response_body.get('outputs', [])
      if outputs:
        return outputs[0].get('text', '')
      return ''

    else:
      raise exceptions.InferenceRuntimeError(
          f'Unknown response format for provider: {model_provider}'
      )

  def _process_single_prompt(
      self, prompt: str, config: dict
  ) -> inference.ScoredOutput:
    """Process a single prompt and return a ScoredOutput."""
    try:
      request_body = self._prepare_request_body(prompt, config)

      response = self._client.invoke_model(
          modelId=self.model_id,
          body=json.dumps(request_body),
          contentType='application/json',
          accept='application/json'
      )

      response_body = json.loads(response['body'].read())
      output_text = self._extract_response_text(response_body)

      # Add markdown fences if requested
      if self.fence_output:
        if self.format_type == data.FormatType.JSON:
          output_text = f'```json\n{output_text}\n```'
        elif self.format_type == data.FormatType.YAML:
          output_text = f'```yaml\n{output_text}\n```'

      return inference.ScoredOutput(score=1.0, output=output_text)

    except Exception as e:
      raise exceptions.InferenceRuntimeError(
          f'Bedrock API error: {str(e)}', original=e
      ) from e

  def infer(
      self, batch_prompts: Sequence[str], **kwargs
  ) -> Iterator[Sequence[inference.ScoredOutput]]:
    """Runs inference on a list of prompts via Bedrock's API.

    Args:
      batch_prompts: A list of string prompts.
      **kwargs: Additional generation params (temperature, top_p, top_k, etc.)

    Yields:
      Lists of ScoredOutputs.
    """
    config = {
        'temperature': kwargs.get('temperature', self.temperature),
    }
    if 'max_output_tokens' in kwargs:
      config['max_output_tokens'] = kwargs['max_output_tokens']
    if 'top_p' in kwargs:
      config['top_p'] = kwargs['top_p']
    if 'top_k' in kwargs:
      config['top_k'] = kwargs['top_k']

    # Use parallel processing for batches larger than 1
    if len(batch_prompts) > 1 and self.max_workers > 1:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=min(self.max_workers, len(batch_prompts))
      ) as executor:
        future_to_index = {
            executor.submit(
                self._process_single_prompt, prompt, config.copy()
            ): i
            for i, prompt in enumerate(batch_prompts)
        }

        results: list[inference.ScoredOutput | None] = [None] * len(
            batch_prompts
        )
        for future in concurrent.futures.as_completed(future_to_index):
          index = future_to_index[future]
          try:
            results[index] = future.result()
          except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'Parallel inference error: {str(e)}', original=e
            ) from e

        for result in results:
          if result is None:
            raise exceptions.InferenceRuntimeError(
                'Failed to process one or more prompts'
            )
          yield [result]
    else:
      # Sequential processing for single prompt or worker
      for prompt in batch_prompts:
        result = self._process_single_prompt(prompt, config.copy())
        yield [result]
