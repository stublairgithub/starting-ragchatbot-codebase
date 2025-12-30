# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) chatbot** for querying course materials. The system uses:
- **ChromaDB** for vector storage with semantic search
- **Anthropic Claude API** with tool-use pattern for intelligent query routing
- **FastAPI** backend serving both API endpoints and static frontend
- **Sentence Transformers** for text embeddings

## Development Commands

**IMPORTANT**: This project uses `uv` as its package manager. **Always use `uv` commands - never use `pip` directly.**

### Setup
```bash
# Install dependencies
uv sync

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application
```bash
# Preferred method
./run.sh

# Manual start (from backend directory)
cd backend && uv run uvicorn app:app --reload --port 8000

# Run any Python script
uv run python script.py
```

Access points:
- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Architecture Overview

### Request Flow (User Query → Response)

1. **Frontend** (`script.js:45-96`) - User submits query via POST to `/api/query` with `session_id`
2. **API Layer** (`app.py:56`) - FastAPI endpoint validates request, calls RAG system
3. **RAG System** (`rag_system.py:102`) - Orchestrator that:
   - Retrieves conversation history from `SessionManager`
   - Passes query to `AIGenerator` with available tools
   - Collects sources from `ToolManager`
   - Saves exchange to session history
4. **AI Generator** (`ai_generator.py:43`) - Claude API interface that:
   - Makes initial API call with tools available
   - If Claude requests tool use (`stop_reason: "tool_use"`), executes search
   - Makes second API call with search results to generate final answer
5. **Search Tool** (`search_tools.py:52`) - Executes via `VectorStore.search()`, formats results, tracks sources
6. **Vector Store** (`vector_store.py:61`) - ChromaDB wrapper that:
   - Converts query to embeddings via Sentence Transformers
   - Performs semantic search with optional course/lesson filtering
   - Returns top 5 most similar chunks with metadata

### Two-Stage Claude API Pattern

The system uses a **tool-use pattern** where Claude makes decisions autonomously:

**Stage 1:** Claude analyzes query and decides whether to search
```python
# First API call with tools available
response = client.messages.create(
    tools=[search_course_content],
    tool_choice={"type": "auto"}
)
# Claude returns: stop_reason="tool_use" OR direct answer
```

**Stage 2:** If tool requested, execute search and call Claude again
```python
# Execute search, build conversation with tool results
messages = [
    {role: "user", content: original_query},
    {role: "assistant", content: [tool_use_block]},
    {role: "user", content: [tool_result_block]}
]
# Second API call generates final answer using search results
```

### Data Models & Storage

**Document Processing Flow:**
1. Course files in `docs/` follow format:
   ```
   Course Title: [title]
   Course Link: [url]
   Course Instructor: [name]

   Lesson 0: [title]
   Lesson Link: [url]
   [content...]
   ```

2. `DocumentProcessor.process_course_document()` (`document_processor.py:97`):
   - Extracts metadata (course title, instructor, lessons)
   - Chunks text by sentences with 800 char limit, 100 char overlap
   - Adds context prefixes: `"Course {title} Lesson {N} content: {chunk}"`
   - Returns `(Course, List[CourseChunk])`

3. Storage in ChromaDB via `VectorStore`:
   - **Collection `course_catalog`**: Course metadata for fuzzy name matching
   - **Collection `course_content`**: Text chunks with embeddings for semantic search
   - Chunks stored with metadata: `{course_title, lesson_number, chunk_index}`

**Core Data Models** (`models.py`):
- `Course`: title (unique ID), instructor, lessons list, course_link
- `Lesson`: lesson_number, title, lesson_link
- `CourseChunk`: content, course_title, lesson_number, chunk_index

### Key Components

**RAGSystem** (`rag_system.py`) - Main orchestrator
- `add_course_document()`: Process single course file
- `add_course_folder()`: Batch load from `docs/`, prevents duplicates
- `query()`: Main query handler - coordinates all components
- `get_course_analytics()`: Returns course count and titles

**AIGenerator** (`ai_generator.py`) - Claude interface
- System prompt instructs Claude to use search tool only for course-specific questions
- Temperature: 0 (deterministic), Max tokens: 800
- `generate_response()`: Main entry point
- `_handle_tool_execution()`: Manages tool call loop

**VectorStore** (`vector_store.py`) - ChromaDB wrapper
- `search()`: Main search interface with course/lesson filtering
- `_resolve_course_name()`: Uses semantic search on catalog for fuzzy matching
- `_build_filter()`: Constructs ChromaDB filters for metadata
- `add_course_metadata()`, `add_course_content()`: Index data

**SearchTools** (`search_tools.py`) - Tool definitions
- `Tool` (ABC): Interface for extensible tool system
- `CourseSearchTool`: Implements course content search
  - Formats results with `[Course - Lesson N]` headers
  - Tracks `last_sources` for UI display
- `ToolManager`: Registers tools, executes by name, manages sources

**SessionManager** (`session_manager.py`) - Conversation state
- Maintains per-session message history (default: last 2 exchanges = 4 messages)
- Methods: `create_session()`, `add_exchange()`, `get_conversation_history()`

### Configuration

All settings in `config.py` (dataclass):
```python
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_RESULTS = 5
MAX_HISTORY = 2  # conversation exchanges
CHROMA_PATH = "./chroma_db"
```

### Important Notes

**Startup Behavior** (`app.py:88`):
- On server start, automatically loads all courses from `../docs`
- Uses `add_course_folder(clear_existing=False)` to prevent re-indexing
- Existing courses checked via `get_existing_course_titles()`

**Session Management:**
- Sessions created on first query if no `session_id` provided
- History stored in memory (not persisted across restarts)
- Limited to `MAX_HISTORY` exchanges to control context size

**ChromaDB Persistence:**
- Data persists in `./backend/chroma_db/` directory
- Collections auto-created on first use via `get_or_create_collection()`
- Use `vector_store.clear_all_data()` to reset database

**Tool Extensibility:**
- Add new tools by implementing `Tool` interface
- Register with `ToolManager.register_tool()`
- Tools automatically available to Claude via `get_tool_definitions()`

**Frontend Architecture:**
- Pure vanilla JavaScript (no framework)
- Markdown rendering via `marked.js`
- Session ID stored in global state, passed with each request
- Sources displayed in collapsible `<details>` element
