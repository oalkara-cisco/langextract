# AWS Bedrock Examples

This directory contains examples for using LangExtract with AWS Bedrock for cloud-based LLM inference.

For setup instructions and documentation, see the [main README's Bedrock section](../../README.md#using-aws-bedrock-models).

## Quick Reference

**Basic setup:**
```bash
# Install with Bedrock support
pip install langextract[bedrock]

# Configure AWS credentials
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile  # optional

# Run the example
python quickstart.py
```

**Advanced usage:**
```bash
# List available model examples
python quickstart.py --list-models

# Use a specific model
python quickstart.py --model-id anthropic.claude-3-sonnet-20240229-v1:0

# Use different region
python quickstart.py --aws-region us-west-2

# Adjust temperature
python quickstart.py --temperature 0.7
```

## Prerequisites

1. **AWS Account**: You need an AWS account with access to Bedrock
2. **AWS Credentials**: Configure using one of these methods:
   - AWS CLI: `aws configure`
   - Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   - IAM roles (if running on AWS infrastructure)
   - AWS profiles: `AWS_PROFILE=your-profile`

3. **Bedrock Permissions**: Your AWS credentials need these permissions:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [{
       "Effect": "Allow",
       "Action": [
         "bedrock:InvokeModel",
         "bedrock:ListFoundationModels"
       ],
       "Resource": "*"
     }]
   }
   ```

4. **Model Access**: Request access to Bedrock models in the AWS console
   - Go to AWS Bedrock console → Model access
   - Request access for the models you want to use

## Supported Models

LangExtract supports all major Bedrock model families:

### Anthropic Claude (Recommended)
- `anthropic.claude-3-haiku-20240307-v1:0` - Fast, cost-effective
- `anthropic.claude-3-sonnet-20240229-v1:0` - Balanced performance
- `anthropic.claude-3-opus-20240229-v1:0` - Highest capability

### Amazon Titan
- `amazon.titan-text-lite-v1` - Lightweight model
- `amazon.titan-text-express-v1` - Enhanced model

### Meta Llama
- `meta.llama3-8b-instruct-v1:0` - 8B parameter model
- `meta.llama3-70b-instruct-v1:0` - 70B parameter model

### Cohere Command
- `cohere.command-text-v14` - Full-featured model
- `cohere.command-light-text-v14` - Lightweight variant

### AI21 Labs Jurassic
- `ai21.j2-mid-v1` - Mid-size model
- `ai21.j2-ultra-v1` - Large model

### Mistral
- `mistral.mistral-7b-instruct-v0:2` - 7B parameter model
- `mistral.mixtral-8x7b-instruct-v0:1` - Mixture of experts model

**Note**: Model availability varies by AWS region. Check the Bedrock console for your region.

## Files

- `quickstart.py` - Basic extraction example with comprehensive error handling
- `README.md` - This documentation file

## Common Issues

### "AccessDeniedException" or "UnauthorizedOperation"
- Check your AWS credentials are configured correctly
- Verify your IAM user/role has Bedrock permissions
- Ensure you've requested model access in the Bedrock console

### "ValidationException" or "ResourceNotFoundException"
- The model may not be available in your region
- Try a different region (us-east-1, us-west-2 typically have most models)
- Use `--list-models` to see example model IDs

### Import Error for boto3
```bash
pip install langextract[bedrock]
```

### Region Not Supported
Bedrock is available in these regions:
- us-east-1 (N. Virginia)
- us-west-2 (Oregon)
- ap-southeast-1 (Singapore)
- ap-northeast-1 (Tokyo)
- eu-west-1 (Ireland)
- eu-central-1 (Frankfurt)

Check AWS documentation for the latest list.

## Model Licenses

Bedrock models come with their own licenses:
- **Anthropic Claude**: [Claude API Terms](https://www.anthropic.com/claude-api-terms)
- **Meta Llama**: [Llama License](https://llama.meta.com/llama-downloads/)
- **Amazon Titan**: [AWS Service Terms](https://aws.amazon.com/service-terms/)

Please review the license for any model you use.
