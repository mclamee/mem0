# Pull Request Guide

## 🎉 Implementation Complete!

Your feature implementation is ready for submission. Here's how to create the PR:

## What Was Implemented

### Code Changes
- **HuggingFaceRerankerConfig** - Added HTTP API support parameters
  - `huggingface_base_url`: TEI service URL
  - `timeout`: Request timeout configuration

- **HuggingFaceReranker** - Dual-mode reranker implementation
  - HTTP API mode for TEI integration
  - Local inference mode (existing functionality)
  - Automatic mode detection based on configuration
  - Error handling and fallback mechanism

### Documentation
- **examples/reranker_tei_example.py** - Complete usage example
- **tests/reranker/test_huggingface_reranker.py** - Comprehensive test suite

## Steps to Create PR

### 1. Fork the Repository (if not done)

Go to https://github.com/mem0ai/mem0 and click "Fork" to create your own copy.

### 2. Add Your Fork as Remote

```bash
# Add your fork as a remote (replace YOUR_USERNAME with your GitHub username)
git remote add fork git@github.com:YOUR_USERNAME/mem0.git

# Or if using HTTPS:
git remote add fork https://github.com/YOUR_USERNAME/mem0.git

# Verify remotes
git remote -v
```

### 3. Push Your Branch

```bash
# Push to your fork
git push -u fork feature/huggingface-reranker-http-api
```

### 4. Create Pull Request

1. Go to https://github.com/mem0ai/mem0
2. You should see a banner "Compare & pull request" for your recently pushed branch
3. Click "Compare & pull request"
4. Fill in the PR details using the template below

## PR Title

```
feat: Add HTTP API support to HuggingFace reranker for TEI integration
```

## PR Description Template

```markdown
## Summary

This PR adds support for HuggingFace Text Embeddings Inference (TEI) HTTP API to the HuggingFaceReranker, enabling high-performance reranking in production environments without loading models locally.

## Motivation

Currently, the HuggingFaceReranker only supports local model inference, which requires:
- Loading large models into memory
- GPU/CPU resources on the application server
- Model initialization time on startup

This limitation makes it challenging to:
- Deploy reranking in serverless/stateless environments
- Scale reranking independently from the application
- Use HuggingFace's official TEI service for optimal performance

## Changes

### Configuration
- Add `huggingface_base_url` parameter to enable HTTP API mode
- Add `timeout` parameter for API request configuration
- Maintain backward compatibility (no breaking changes)

### Implementation
- **Dual-mode support**: Automatically detects HTTP API vs local inference mode
- **HTTP API integration**: Calls TEI `/rerank` endpoint with proper error handling
- **Code refactoring**: Separate `_rerank_http_api()` and `_rerank_local()` methods
- **Error handling**: Graceful fallback to original order if API fails
- **Logging**: Add informative logs for debugging and monitoring

### Documentation
- Comprehensive usage example with both modes
- Extensive test coverage for HTTP API functionality
- Clear configuration examples

## Usage Example

```python
from mem0 import Memory

# HTTP API mode (new feature)
config = {
    "reranker": {
        "provider": "huggingface",
        "config": {
            "huggingface_base_url": "http://reranker-service:8080",
            "timeout": 30,
            "top_k": 5
        }
    }
}

# Local mode (existing functionality - unchanged)
config = {
    "reranker": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-reranker-base",
            "device": "cuda"
        }
    }
}
```

## Deployment Example

```bash
# Deploy TEI service
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.6 \
  --model-id BAAI/bge-reranker-large
```

## Testing

All tests pass:
```bash
pytest tests/reranker/test_huggingface_reranker.py -v
```

Test coverage includes:
- ✅ HTTP API mode initialization
- ✅ Successful reranking via TEI API
- ✅ Error handling and fallback
- ✅ Multiple document field formats (memory/text/content)
- ✅ Empty document handling
- ✅ Configuration validation

## Backward Compatibility

✅ **Fully backward compatible**
- Existing code continues to work without changes
- Local inference mode is still the default
- HTTP API mode only activates when `huggingface_base_url` is provided

## Dependencies

- Requires `httpx` for HTTP API mode only
- No new dependencies for local inference mode
- Graceful error message if httpx is missing

## Related

- HuggingFace TEI: https://github.com/huggingface/text-embeddings-inference
- Similar pattern used in HuggingFaceEmbedding (mem0/embeddings/huggingface.py)

## Checklist

- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] Code is commented, particularly in complex areas
- [x] Documentation updated (examples added)
- [x] Tests added for new functionality
- [x] All tests pass locally
- [x] No breaking changes introduced
- [x] Feature is backward compatible

## Screenshots / Logs

Example output showing HTTP API mode:
```
INFO:mem0.reranker.huggingface_reranker:Initialized HuggingFaceReranker in HTTP API mode with URL: http://localhost:8080/rerank
```

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

## Additional Notes

### Why This Approach?

1. **Consistency**: Follows the same pattern as HuggingFaceEmbedding
2. **TEI Official Support**: Uses HuggingFace's official inference server
3. **Production Ready**: Includes timeout, error handling, and logging
4. **Developer Friendly**: Clear examples and comprehensive tests

### Benefits

- **Performance**: TEI provides optimized inference
- **Scalability**: Separate reranking service can scale independently
- **Flexibility**: Works with local deployment or Kubernetes services
- **Cost**: Can use smaller application servers, offload GPU to dedicated service

### Future Enhancements (optional)

Potential follow-up PRs could include:
- API key authentication support
- Retry logic with exponential backoff
- Batch size configuration for HTTP mode
- Metrics and telemetry

## Questions?

If reviewers have questions about:
- Implementation choices
- Test coverage
- Documentation
- Configuration design

Please comment on the PR and I'll address them promptly!
```
