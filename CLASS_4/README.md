## Simple RAG Demo (CLASS_4)

### Pre-requirements
1. Install Ollama from the official website: [https://ollama.com/](https://ollama.com/)
2. Download the required models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

3. Install the Ollama Python library, Google GenAI library and HF SentenceTransformers:

```bash
pip install ollama
pip install -q -U google-genai
pip install -U sentence-transformers

```

This exercise was created based on following Hugging Face tutorial: [Code a simple RAG from scratch](https://huggingface.co/blog/ngxson/make-your-own-rag).

### Useful resources
- [Sentece Transformers](https://huggingface.co/sentence-transformers)
- [Gemini API quickstart](https://ai.google.dev/gemini-api/docs/quickstart)