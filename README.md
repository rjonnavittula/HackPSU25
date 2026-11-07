# HackPSU25 — Rachel, a Local AI Assistant

A Python command-line AI assistant built around local Ollama inference, persistent conversation history, retrieval of relevant past chats, and a small tool-calling interface.

## What is implemented

- **Local inference:** uses `llama3.2` through Ollama to generate responses.
- **Conversation memory:** saves sessions, titles, timestamps, and summaries as local JSON files.
- **Semantic retrieval:** embeds session summaries using `nomic-embed-text`, scores cosine similarity, and adds relevant prior-chat summaries to the current conversation.
- **Tool calling:** parses structured tool-call JSON and supports time lookup and searching saved notes.
- **Session controls:** enter `history`, `switch <chat-id>`, `new chat`, or `exit` in the terminal.

## Requirements

- Python 3
- [Ollama](https://ollama.com/) running locally
- Ollama Python package

Download the expected models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Install the Python dependency:

```bash
python -m pip install ollama
```

## Run

```bash
python assistant.py
```

The application creates a `sessions/` directory for locally saved conversations. Do not commit personal session data.

## Structure

```text
assistant.py    # Conversation loop, persistence, retrieval, and tool calling
sessions/       # Generated locally at runtime (not included in this repository)
```

## Scope and limitations

This is an experimental command-line prototype, not a production-ready agent or a security boundary. It uses JSON files rather than a vector database, retrieves from previously saved chat summaries, and exposes only the tools defined in `assistant.py`. It does not include a web frontend or hosted deployment.

## Background

Developed as a HackPSU-era personal AI assistant experiment. The project reflects interests in local models, agent memory, retrieval-augmented generation, and tool-use workflows.
