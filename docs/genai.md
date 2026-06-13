# Generative AI

## 1. LangChain

### The Problem It Solves

Building applications on top of Large Language Models (LLMs) involves repetitive, low-level work: constructing prompt strings, managing API calls to different providers, parsing unstructured text responses, and stitching together retrieval logic. Every new project requires re-implementing these same patterns from scratch.

LangChain is a framework (a pre-built toolkit of reusable code) that provides standardised abstractions for all of these concerns, letting developers focus on application logic rather than integration plumbing ("plumbing" here means the tedious wiring code that connects different systems together — not your core logic, but the glue between components).

### Core Abstractions

**Prompt Templates** allow you to define reusable prompt structures with placeholder variables. Instead of building prompt strings via f-strings or concatenation, you declare a template once and fill it with different inputs at runtime. This separates prompt engineering from application code.

**LLM Wrappers** provide a unified interface across model providers (OpenAI, Azure OpenAI, Google Gemini, Anthropic, open-source models via Ollama, etc.). You can swap providers by changing one line of configuration without rewriting your application logic.

**Chains** are the fundamental composition primitive. A chain takes the output of one step and feeds it as input to the next. For example: format a prompt → call the LLM → parse the output. Chains are sequential pipelines where each step transforms data and passes it forward.

**LCEL (LangChain Expression Language)** is the modern syntax for composing chains using the `|` (pipe) operator. It replaces older verbose chain classes with a concise, readable pipeline notation:

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"topic": "quantum computing"})
```

LCEL chains are lazy — "lazy" means they describe the computation graph without executing it immediately. Nothing actually runs until you explicitly call `.invoke()` (run once), `.stream()` (get results token-by-token as they're generated), or `.batch()` (run the same chain on multiple inputs in parallel). They also provide built-in support for async execution (non-blocking — your program can do other work while waiting for the LLM response instead of freezing).

**Output Parsers** convert raw LLM text responses into structured data. Since LLMs produce unstructured text by default (just a string of words), parsers handle extracting JSON, lists, Pydantic objects (Pydantic is a Python library for defining data shapes with type validation — e.g., ensuring a "name" field is a string and an "age" field is an integer), or other formats from the model's output. They can also inject formatting instructions into the prompt automatically.

**Retrievers** connect LLMs to external knowledge sources. Rather than relying solely on the model's training data, a retriever fetches relevant documents from a vector store, database, or search engine, which are then included in the prompt context. This is the foundation of RAG (covered in detail below).

**Agents** give the LLM the ability to decide which tools to call and in what order. Rather than following a fixed chain, the LLM reasons about what action to take, executes it, observes the result, and decides the next step. Early LangChain agents were functional but brittle — they lacked structured state management and reliable looping, which motivated the creation of LangGraph.

### The Architectural Limitation

LangChain chains follow a DAG (Directed Acyclic Graph) structure. A DAG is a graph where connections between nodes only go forward — like a one-way street system with no U-turns. "Directed" means edges have a direction (A→B, not just A—B). "Acyclic" means no cycles/loops — you can never follow edges and end up back where you started. Data flows forward through a sequence of steps, potentially branching into parallel paths, but never looping back. There is no native mechanism for:

- Re-running a step based on its own output
- Maintaining shared mutable state that multiple steps can read and write
- Conditional branching that routes execution to entirely different paths

This makes LangChain ideal for linear pipelines but insufficient for complex agentic workflows that require iteration, decision-making, and memory.


### Is a Diamond Pattern Valid in a DAG?

Yes. The "acyclic" constraint only prohibits loops — branching out and merging back is perfectly valid because data still only flows forward.

```
        ┌──── branch A ────┐
input ──┤                  ├──── output parser ──── final output
        └──── branch B ────┘
```

In LCEL, there are two ways to express this:

**`RunnableParallel`** — both branches run simultaneously, outputs merge as a dict:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    path_a=chain_a,
    path_b=chain_b
)
# output is {"path_a": ..., "path_b": ...}
diamond_chain = parallel | merge_and_parse
```

**`RunnableBranch`** — mutually exclusive routing, only one branch runs based on a condition:

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["type"] == "A", chain_a),
    chain_b  # default
)
diamond_chain = branch | output_parser
```

The key distinction: `RunnableParallel` runs all branches and merges results; `RunnableBranch` routes to exactly one branch. Both are valid DAGs. What breaks the DAG model is a loop where a node's output feeds back into a prior node — which is exactly what LangGraph adds.

### Can't I Just Use RunnableBranch for Tool Fallbacks?

Yes — for a **fixed, known number of fallbacks**. `tool_a → evaluate → RunnableBranch(sufficient → output_parser, insufficient → tool_b → output_parser)` is a valid DAG.

It breaks down when the number of iterations isn't known upfront. If `tool_b` is also insufficient, you'd need to nest another branch inside, and another inside that. A DAG must be **fully defined before execution starts** — every node and edge must exist before `.invoke()` is called. You can't encode "keep trying until it works" in a static graph.

LangGraph solves this because the routing decision (loop again vs. exit) is made at *runtime* by a conditional edge function inspecting the current state — the graph doesn't need to know how many iterations it will take.

---

## 2. LangGraph

### The Problem It Solves

Many real-world LLM applications are not simple pipelines. An AI agent might need to: call a tool, evaluate the result, decide the result is insufficient, call a different tool, evaluate again, and repeat until satisfied. A multi-agent system might need several agents to collaborate, each reading and updating a shared workspace. A workflow might need to pause for human review before continuing.

None of these patterns fit cleanly into a DAG. LangGraph is a framework (built on top of LangChain) that models LLM applications as stateful, cyclical graphs — enabling loops, conditional routing, shared state, and persistence as first-class features.

### Core Concepts

**Nodes** are the individual units of computation. Each node is a Python function that receives the current state, performs some work (calling an LLM, executing a tool, transforming data), and returns an updated state. Nodes are where your actual logic lives.

**Edges** define how control flows between nodes. There are two types:
- *Fixed edges* — unconditional connections (after node A, always go to node B)
- *Conditional edges* — a routing function examines the current state and decides which node to transition to next

**State** is a typed object (typically a TypedDict or Pydantic model — both are ways of defining a Python dictionary/object where each field has a declared type, so you get autocomplete and error checking) available to all nodes in the graph. A node reads a state snapshot and returns only its updates; LangGraph applies those updates using per-field reducers. This is not a freely mutable global dictionary, but runtime-managed state that accumulates as the graph executes.

**Cycles** are what make LangGraph a graph rather than a DAG. A node can route back to a previous node, creating a loop. This is essential for agent behaviour: reason → act → observe → reason again. The loop continues until a termination condition is met (e.g., the agent decides it has a final answer).

### Why Didn't LangChain Use Global State?

It could: applications already used memory, agent scratchpads, or dictionaries passed through chains. LCEL deliberately favoured explicit `input → output` transformations because each invocation remains isolated and its data dependencies stay visible.

A freely mutable global object would create problems when batching, running asynchronously, or executing parallel branches: multiple tasks could overwrite each other's values, observe half-completed updates, or require locks that remove parallelism. It would also make streaming harder because state mutation does not define chunk ordering, completion, backpressure, or which invocation owns an update.

LangGraph formalises the useful version of shared state. Parallel nodes read the same snapshot and return partial updates, which the runtime merges at a controlled boundary using declared reducers. Therefore, common state itself was not the anti-pattern; hidden, uncontrolled shared mutation was.

### Why Not Just Use a Python Loop Around a Chain?

Technically, you could wrap a LangChain chain in a `while` loop and achieve iteration. However, this approach breaks down quickly:

- **State management becomes manual** — you must explicitly track what carries over between iterations, handle edge cases, and ensure consistency.
- **No conditional routing** — branching logic gets buried in if/else blocks that are hard to reason about.
- **No observability** — there's no built-in way to inspect intermediate states, trace execution paths, or debug failures at specific steps.
- **No persistence** — if the process crashes, all intermediate work is lost.
- **No human-in-the-loop** — there's no natural place to pause and resume execution.
- **Doesn't scale to multi-agent** — coordinating multiple agents with shared state in raw Python becomes unmaintainable.

LangGraph provides all of these as infrastructure. The graph structure is declarative (you describe *what* the workflow looks like — which nodes exist and how they connect — rather than writing imperative step-by-step control flow), observable (you can inspect what's happening at each step), and can be interrupted and resumed at any point.

### The ReAct Pattern

ReAct (Reasoning + Acting) is the dominant pattern for LLM agents. The cycle works as follows:

1. **Reason** — The LLM examines the current state and decides what to do next
2. **Act** — The chosen action is executed (calling a tool, querying a database, etc.)
3. **Observe** — The result of the action is added to the state
4. **Repeat** — The LLM re-examines the updated state and decides whether to act again or produce a final answer

This is inherently a loop, and LangGraph's cyclical graph structure maps directly to it. The LLM node routes to an action node, which routes back to the LLM node, until the LLM routes to the "end" node.

### When to Use LangChain vs LangGraph

| Scenario | Recommended Tool |
|---|---|
| Single-pass RAG (retrieve → augment → generate) | LangChain / LCEL |
| Summarisation, translation, simple transformations | LangChain / LCEL |
| Agent that iteratively uses tools until task is complete | LangGraph |
| Workflow requiring human approval at certain steps | LangGraph |
| Multiple agents collaborating on a task | LangGraph |
| Complex branching logic (different paths for different inputs) | LangGraph |
| Any workflow needing persistence or resumability | LangGraph |

These tools are complementary, not competing. In practice, LangGraph orchestrates the overall flow (deciding what runs when, managing state, handling loops), while LangChain components (prompt templates, LLM wrappers, retrievers, output parsers) are used inside individual nodes to do the actual work.

---

## 3. Checkpointers (State Persistence in LangGraph)

### What a Checkpointer Does

A checkpointer is a persistence layer that automatically saves the graph's state after every node execution. It serialises (converts the in-memory Python object into a storable format like JSON or bytes) the state object and stores it in a backend (memory, SQLite, PostgreSQL, etc.), keyed by a thread identifier.

### Why Persistence Matters

**Resumability** — If a process crashes, a server restarts, or a user closes their browser, the graph can resume from the last completed step rather than starting over. This is essential for long-running workflows.

**Conversation Memory** — For chatbot applications, the checkpointer stores the conversation history. When the same user sends a new message (using the same `thread_id`), the graph loads the full prior state and continues naturally.

**Human-in-the-Loop** — A graph can be designed to pause at specific nodes (e.g., before executing a sensitive action). The state is saved, the system waits for human approval (which may come minutes, hours, or days later), and then resumes from exactly where it stopped.

**Time Travel and Debugging** — Since state is saved at every step, you can replay execution from any checkpoint. This is invaluable for debugging: you can inspect the state at the exact point where something went wrong, or re-run from that point with different inputs.

### How It Works in Practice

Every invocation of a LangGraph graph requires a `thread_id` in its configuration:

```python
config = {"configurable": {"thread_id": "user-123-session-1"}}
result = graph.invoke(input, config)
```

The checkpointer stores a snapshot of the state after each node completes, associated with this `thread_id`. On subsequent invocations with the same `thread_id`, the graph loads the most recent checkpoint and continues from there rather than starting fresh.

### Available Checkpointer Implementations

| Checkpointer | Use Case |
|---|---|
| `MemorySaver` | Development and testing. State lives in-memory and is lost on process restart. |
| `SqliteSaver` | Local persistence for single-user applications or prototyping. |
| `PostgresSaver` | Production deployments with concurrent users, durability requirements, and horizontal scaling. |

---

## 4. Retrieval-Augmented Generation (RAG)

### The Problem It Solves

LLMs are trained on a fixed dataset with a knowledge cutoff date. They cannot access private documents, real-time information, or domain-specific knowledge that wasn't in their training data. RAG bridges this gap by retrieving relevant external information at query time and injecting it into the prompt.

### How RAG Works

The RAG pipeline has two phases:

**Indexing (offline, done once or periodically):**
1. Load documents from sources (files, databases, APIs, web pages)
2. Split documents into smaller chunks (because LLMs have context length limits and smaller chunks improve retrieval precision)
3. Generate embedding vectors for each chunk using an embedding model. An embedding is a list of numbers (e.g., 1536 floats) that represents the *meaning* of a piece of text. Texts with similar meaning get similar numbers, so you can find related content by comparing these number lists mathematically.
4. Store the chunks and their vectors in a vector store (a specialised database designed for fast similarity search over these number lists — e.g., Pinecone, Chroma, FAISS, Weaviate)

**Retrieval and Generation (online, per query):**
1. User submits a question
2. The question is embedded using the same embedding model
3. A similarity search (comparing how close two embedding vectors are — typically using cosine similarity, which measures the angle between two vectors; smaller angle = more similar meaning) in the vector store finds the most relevant chunks
4. The retrieved chunks are inserted into the prompt alongside the user's question
5. The LLM generates a response grounded in the retrieved context

### Why Chunking Matters

If you embed an entire 50-page document as one vector, it must represent many unrelated topics at once. The meaning of one useful sentence gets weakened by all the other content. This is called **semantic dilution**: the vector becomes a vague average of the document and may not closely match a specific question. Smaller chunks keep each embedding focused and improve retrieval precision.

**Who creates the chunks?** The developer chooses a strategy and configuration; a text-splitting library performs the split. Common strategies are fixed-size, sentence/paragraph-aware, semantic, and structure-aware splitting. The correct size is determined by testing retrieval with representative questions, not by a universal rule.

**Small versus large chunks:**

| Small chunks | Large chunks |
|---|---|
| Match focused questions precisely | Preserve more surrounding context |
| Add less irrelevant text to the prompt | Keep related sentences, headings, and definitions together |
| May separate a statement from its heading, subject, or definition | Cause semantic dilution by mixing several topics into one vector |
| May require several chunks to answer one question | Consume more of the LLM's context window |

Overlap can preserve information at boundaries, but excessive overlap creates duplicates. A common starting point is 300–800 tokens with 10–20% overlap, followed by evaluation.

**Parent-child retrieval** lets the system search small pieces but give the LLM a larger section. For example, an `Annual Leave Policy` parent section may contain these child chunks:

```text
Child 1: Employees receive 20 vacation days.
Child 2: Contractors do not receive paid vacation.
Child 3: Unused vacation expires at year-end.
```

For `What happens to unused vacation?`, vector search precisely finds Child 3. Its stored `parent_id` is then used to fetch the complete `Annual Leave Policy`, which gives the LLM the surrounding context:

```text
search small child → find exact match → return larger parent
```

### Handling Large Documents

Treat a large document as a hierarchy instead of splitting it blindly every fixed number of tokens:

```text
document → headings/sections → paragraphs or child chunks
```

First extract structural elements such as headings, paragraphs, tables, and page numbers. Use sections as parent chunks, split oversized sections into smaller searchable children, and store metadata such as `document_id`, `parent_id`, heading, and page with each child. Search the children and fetch their parent when more context is needed.

Large document collections use the same chunking approach, but additionally require batch ingestion, deduplication, and metadata filters such as department, document type, tenant, or year to reduce the search space.

### How an Embedding Model Creates a Vector

```text
text → tokens → token IDs → token vectors → transformer → pooling → text vector
```

1. **Tokenisation:** The tokenizer splits text into vocabulary entries. For example, it might split `Employees receive vacation` into `Employ`, `ees`, ` receive`, and ` vacation`.
2. **Token IDs:** Each vocabulary entry has an arbitrary integer identifier, such as `[8321, 291, 4256, 10982]` (illustrative values). These numbers are lookup keys, not meanings or embeddings: `8321` simply identifies one row in the tokenizer's vocabulary.
3. **Token vectors:** Each ID selects a learned row from the model's embedding matrix. With a 50,000-token vocabulary and 768-dimensional internal vectors, this matrix has shape `50,000 × 768`.
4. **Contextualisation:** Transformer layers use self-attention to update each token vector using the surrounding tokens. This lets words such as `bank` represent different meanings in financial and river contexts.
5. **Pooling and normalisation:** Pooling combines all contextual token vectors into one vector for the complete chunk. With **mean pooling**, each vector position is averaged independently:

```text
Employees → [0.2, 0.4, 0.6]
receive   → [0.4, 0.1, 0.3]
vacation  → [0.6, 0.4, 0.0]

mean      → [(0.2+0.4+0.6)/3, (0.4+0.1+0.4)/3, (0.6+0.3+0.0)/3]
          → [0.4, 0.3, 0.3]
```

Real token vectors may have hundreds or thousands of positions, but the same operation is applied to every position. Other models use a special token, such as `[CLS]`, whose vector gathers information from all tokens through attention, or use learned weights so important tokens contribute more. The model or embedding API normally performs pooling internally and returns only the final text vector. It may then normalise that vector for similarity comparison.

Embedding models are commonly trained with related and unrelated text pairs. Training moves a query closer to its relevant passage and farther from irrelevant passages. Meaning is therefore distributed across the complete vector; an individual dimension usually has no simple human-readable interpretation.

**What is stored?** A vector-store record commonly contains an `id`, embedding vector, original chunk, and metadata such as document, page, or section. Alternatively, the vector store may contain only `id + vector + metadata`, while the chunk is kept in a separate document or key-value store. In both cases the relationship is `vector ↔ chunk ID ↔ text`; similarity search finds the vector, but the corresponding text is sent to the LLM.

### Why RAG Over Fine-Tuning

| Concern | RAG | Fine-Tuning |
|---|---|---|
| Knowledge updates | Instant — update the vector store | Requires retraining |
| Cost | Low — no GPU training needed | High — training compute costs |
| Hallucination control | High — model has source text to quote | Lower — model may still confabulate ("confabulate" = confidently generate false information that sounds plausible) |
| Transparency | Can show source documents to user | Opaque — no attribution possible |
| Domain adaptation | Limited to what's in retrieved context | Can deeply reshape model behaviour |

RAG is the preferred approach when you need the model to answer questions over a specific corpus of documents. Fine-tuning is preferred when you need to change the model's fundamental behaviour, tone, or reasoning patterns.

---

## 5. Observability — LangSmith vs LangFuse

### Why Observability Matters for LLM Applications

LLM applications are non-deterministic (given the same input, they don't always produce the same output — unlike a normal function where `add(2,3)` always returns 5). Chains and agents involve multiple steps, each of which can fail or produce unexpected results. Without observability tooling, debugging is nearly impossible — you cannot see what prompt was actually sent, what the model returned at each step, how long each step took, or where things went wrong.

### What Both Tools Provide

- **Tracing** — Full visibility into every step of a chain/graph execution: inputs, outputs, latency (how long each step took), token usage (tokens are the units LLMs count — roughly ¾ of a word — and you pay per token), errors
- **Prompt Management** — Version control for prompts (track changes over time, roll back to previous versions), A/B testing (send some traffic to prompt version A, some to B, compare quality), iteration without code changes
- **Evaluation** — Run test datasets against your pipeline and score results (automatically via another LLM judging quality, or with human reviewers)
- **Monitoring** — Track cost, latency, error rates, and quality metrics over time in production

### LangSmith

Developed by the LangChain team. Offers the tightest integration with LangChain and LangGraph — traces are captured automatically with minimal configuration. Provides a polished UI for exploring traces, managing datasets, and running evaluations.

**Trade-off:** Cloud-only. All trace data (which includes your prompts, user inputs, and model outputs) is sent to LangChain's servers. There is no self-hosted option.

### LangFuse

An open-source, framework-agnostic (not tied to any specific framework — works with anything) observability platform. Works with LangChain, LlamaIndex, raw OpenAI calls, or any other framework via its SDK (Software Development Kit — a library you import into your code to interact with a service). Provides similar capabilities to LangSmith (tracing, evaluation, prompt management).

**Key advantage:** Fully self-hostable (you run it on your own servers instead of using someone else's cloud). You can deploy LangFuse on your own infrastructure, meaning sensitive data (prompts, user inputs, model responses) never leaves your network. This is often a hard requirement in regulated industries (finance, healthcare, government, legal) due to data residency laws (regulations requiring data to stay within a certain country or network).

### Decision Guide

| Consideration | Choose LangSmith | Choose LangFuse |
|---|---|---|
| Data residency requirements | No strict requirements | Must keep data on-premises |
| Framework | Exclusively LangChain/LangGraph | Mixed or non-LangChain stack |
| Setup preference | Managed cloud, minimal ops | Willing to self-host |
| Budget | Free tier sufficient or paid plan acceptable | Need unlimited usage (self-hosted = free) |

---

## 6. LLMs — Foundations, Training & Limitations

### What else falls under Generative AI besides LLMs?

Generative AI is any model that **generates new content**. LLMs generate text. They are one modality under the GenAI umbrella. The others:

- **Image generation** — Diffusion models (Stable Diffusion, DALL-E, Midjourney). These use an entirely different architecture from LLMs — they learn to gradually remove noise from random static until a coherent image emerges, guided by a text description.
- **Video generation** — Sora, Runway. Generate video frames conditioned on text or images. Built on diffusion or transformer architectures adapted for temporal sequences.
- **Audio/Music generation** — Suno, MusicLM. Generate sound waveforms or musical sequences from text prompts.
- **Speech synthesis (TTS)** — ElevenLabs, Azure Neural TTS. Generate realistic human voice audio from text input.
- **Multimodal models** — GPT-4o, Gemini. Handle text + image + audio together in one model. These overlap with LLMs but are broader — they can both understand and generate across modalities.
- **Protein/molecular generation** — AlphaFold. Generates 3D molecular structures from amino acid sequences. GenAI applied to biology rather than language.

The unifying principle: all of these learned statistical patterns from massive data and can generate new instances that follow those patterns. LLMs do it for language. Others do it for their respective modality (images, sound, video, molecules).

---

### Why do we say LLMs "think"?

The honest answer first: they probably don't think in the way humans do. Calling it "thinking" is anthropomorphisation (attributing human qualities to non-human things). But here's what makes it *look* like thinking — and why the distinction is less clear-cut than you'd expect.

**What's actually happening mechanically:** The model predicts the next token, one at a time, based on everything that came before it. That's the entire mechanism. There is no separate "reasoning module" or "understanding engine" inside.

**Why this produces something that resembles thinking:** When you train on essentially all of human written knowledge — books, research papers, logical arguments, code, mathematical proofs, debates, stories — you're not just learning which words follow which. You're learning the **structure of how humans reason**. Every cause-effect relationship, every logical deduction, every step in a proof that was ever written down gets compressed into the model's weights (the numbers that define the model's behaviour).

**The baby analogy:** A baby doesn't memorise individual sentences and replay them. It observes patterns — objects, actions, consequences, grammar structure — across thousands of interactions. At some point, it starts generating grammatically correct sentences it has never heard before. Something emerges beyond pure imitation. LLMs undergo a similar process at vastly larger scale: they observe patterns across trillions of tokens and develop internal representations that generalise beyond the specific examples seen.

To be concrete: yes, a baby's first words ("mama", "dada") are directly imitated. But by age 2–3, children routinely produce sentences no one has ever said to them. A classic example from linguistics research: children say things like "I goed to the park" or "two mouses." No adult says "goed" or "mouses" — these aren't imitations. The child has *inferred the rule* (past tense = add "-ed", plural = add "-s") and applied it to words where it shouldn't apply. This is called **overgeneralisation** — it's proof the child learned an abstract pattern, not just memorised specific phrases. They're generating novel combinations from internalised structure. Another example: a 3-year-old who has heard "the dog chased the cat" might say "the cat chased the butterfly" — a perfectly grammatical sentence combining known words in a new configuration they were never explicitly taught. The creativity isn't in the individual words (those are learned) but in the *novel combinations* assembled from internalised rules. LLMs do the same thing — they internalise patterns and generate novel combinations of those patterns that weren't in the training data.

**The key concept — Emergence:** At sufficient scale (billions of parameters, trillions of training tokens), capabilities appear that were never explicitly trained. LLMs can do arithmetic, write working code, reason about hypothetical scenarios, and pass medical licensing exams — yet none of these were the training objective. The only objective was "predict the next token." The reasoning-like behaviour *emerged* from compressing that much structured human knowledge.

**Is it thinking?** We genuinely don't know. What we know is that the compression of all that human reasoning into a model creates something that *behaves* like thinking across a wide range of tasks. Whether there's genuine "understanding" inside, or just extraordinarily sophisticated pattern matching that's functionally indistinguishable from understanding — that's an open philosophical and scientific question.

**Clarification — what we DO understand vs what we DON'T:**

Think of it like a recipe vs a restaurant. We wrote the recipe (the code, the math, the training procedure). We know every ingredient and every step. That's fully understood — nothing mysterious there.

But when you train a model on trillions of words, something happens inside those billions of numbers that we can't easily inspect. It's like: you followed a recipe, but the dish came out way better than you expected, and you're not sure *which* combination of spices made it so good. You know the ingredients. You just can't pinpoint which interactions between them produced the result.

More concretely:
- **What we understand:** The math. How tokens go in, get multiplied through layers of numbers, and a prediction comes out. It's addition and multiplication at the end of the day. No mystery.
- **What we don't understand:** *What* those billions of numbers have collectively learned to represent. We can't open the model and say "ah, this group of numbers handles logic, this group handles French, this group handles sarcasm." It's all tangled together.

This is called the **interpretability problem**. Researchers are working on it — they've found some individual neurons that light up for specific concepts (like one that activates for the Golden Gate Bridge). But explaining the *whole* model? That's like trying to understand a city by looking at every single brick. We know what bricks are. We just can't explain the city from that level.

**Bottom line:** We built the machine. We know how it runs. We just can't fully explain why it's as capable as it is — because the capability comes from scale (billions of numbers interacting), not from any single clever design choice we can point to.

---

### How a transformer turns tokens into a prediction

Consider the prompt `The cat sat on the`.

**1. Tokens and IDs**

The tokenizer splits the text into tokens and maps each token to an integer ID. The ID is only a lookup key in the fixed vocabulary; it does not contain meaning. This vocabulary mapping is separate from the **KV cache**, which stores attention calculations during generation.

**2. Token vectors and model weights**

Each ID selects one row from a learned **embedding matrix**. That row is the token's starting vector. All token vectors have the same length, called the model's hidden size: for example, 4,096 numbers.

The embedding numbers start mostly random and are learned during training. They are only a small part of the model's weights. Most of the billions of weights are in the attention, feed-forward, and output matrices shared by every token.

**3. Positional information**

Attention alone does not know token order, so the model also supplies position information. Otherwise, `dog bites man` and `man bites dog` would look like the same collection of tokens.

Older transformers add a position vector to each token vector. Many modern LLMs use **RoPE**, which rotates the attention query and key vectors according to position. Both approaches tell the model where tokens occur and how far apart they are.

**4. Transformer blocks**

Each block has two main operations:

- **Attention:** each token gathers useful information from earlier tokens. Queries mean "what am I looking for?", keys mean "what do I offer?", and values contain the information to copy. A causal mask prevents a token from seeing future tokens.
- **Feed-forward:** the same neural network processes each token vector separately, refining the information attention gathered.

In simple terms: attention shares information between tokens; feed-forward processes that information. Repeating both operations creates increasingly context-aware vectors.

**5. The number of vectors does not change inside the model**

Five input tokens produce five vectors. Every transformer block still outputs five vectors of the same size; only the numbers inside them change.

```text
The | cat | sat | on | the       <- 5 initial vectors
             transformer blocks
The | cat | sat | on | the       <- 5 contextual vectors
```

There is normally no vector for the blank in `The cat sat on the ___`. The final vector belongs to the last real token, `the`, but now also contains information gathered from the preceding context.

**6. From the last vector to a word**

An output matrix converts the last vector into one raw score, or **logit**, for every token in the vocabulary. **Softmax** converts those scores into probabilities:

```text
mat    65%
floor  12%
chair   5%
...
```

The model selects a token, appends it to the input, and repeats the process to generate the following token. The token count increases between generation steps, not while passing through the transformer blocks. A **KV cache** avoids recalculating keys and values for earlier tokens.

**7. What training changes**

During training, the model predicts the next token at every position in known text. Its probabilities are compared with the actual next tokens, producing an error. Backpropagation then makes small updates to the embeddings and the shared attention, feed-forward, and output weights.

Early in training the probabilities are close to random. After many updates, contexts such as `The cat sat on the` give higher probability to tokens such as `mat`. One training run contains many batches and usually billions or trillions of token predictions, not just one sentence.

**Why attention replaced older sequential models**

RNNs and LSTMs read tokens one at a time and carried a hidden-state summary forward. They did not see only the last word, but distant information had to survive inside that repeatedly updated summary and could fade.

Transformers let each token connect more directly to relevant earlier tokens and process all training positions in parallel. Generation remains sequential because each new output token depends on the tokens already generated.

---

### How does the model pick the next token? Why isn't the output always the same?

The final layer of a transformer produces a probability distribution over the *entire vocabulary* — at each step, the model gives a probability to every possible next token. The question is: how do you pick one?

The obvious answer ("always pick the highest probability one") is called **greedy decoding**, and it's usually a bad idea. The output becomes repetitive, boring, and often degenerate — the model gets stuck in loops, repeating the same phrase. Introducing controlled randomness produces varied, more natural-sounding text.

This is controlled by **sampling strategies**:

**Temperature** — A knob that scales the probability distribution before sampling.

- `temperature = 0` → always pick the highest probability token. Deterministic. Same input → same output every time. Used when you want consistency (e.g., extracting structured data).
- `temperature = 1` → sample from the model's raw distribution. Natural balance.
- `temperature = 2` → flatten the distribution (low-probability tokens get a bigger chance). More creative, more random, more likely to go off the rails.

**Top-k sampling** — Only consider the top `k` most probable tokens, ignore the rest. Sample from those.

**Top-p (nucleus) sampling** — Consider the smallest set of tokens whose cumulative probability ≥ `p` (e.g., 0.9). Adaptive — sometimes that's 3 tokens, sometimes 30, depending on how confident the model is.

**Why even bother with randomness?**

1. **Creativity** — for writing, brainstorming, dialogue, you want variation.
2. **Avoiding loops** — greedy decoding often gets stuck repeating itself because the model's strong prediction reinforces its own previous output.
3. **Exploration** — sometimes the highest-probability token is locally good but leads to a dead end. Picking the second-best occasionally lets the model find better overall completions.

**Practical defaults:**

- Code generation, structured extraction → `temperature = 0` (you want predictability)
- Chat, creative writing → `temperature = 0.7–1.0` (natural variation)
- Brainstorming, idea generation → `temperature = 1.0–1.3` (more diverse outputs)

This is why ChatGPT gives you slightly different answers each time even for the same question — it's sampling, not greedy decoding.

---

### Then what are "reasoning models"? How is that different from normal models?

If all models just predict the next token, what's special about models marketed as "reasoning" or "thinking" models (OpenAI's o1/o3/o4, DeepSeek-R1, etc.) vs standard chat models (GPT-4o, Claude Sonnet)?

**The core insight: it's still next-token prediction. The difference is *what* tokens it's trained to produce.**

A standard chat model is trained to produce the **answer directly**. Ask it "what's 17 × 24?" and it's optimised to immediately output "408."

A reasoning model is trained to produce a **long chain of intermediate thinking tokens before the answer**. Ask it the same question and it generates something like: "Let me break this down. 17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408." Those intermediate tokens aren't decoration — they serve as **working memory**.

**Why intermediate tokens matter mechanically:**

LLMs have no scratchpad, no RAM, no internal workspace. The *only* memory available to them during generation is the sequence of tokens already produced. Each new token is predicted based on everything before it.

This means: if a problem requires 5 logical steps, and you force the model to jump straight to the answer in one token, it has to do all 5 steps of reasoning in a single forward pass through the network. That's often too much — the network isn't deep enough to compute all those steps internally.

But if you let the model write out intermediate steps as tokens, each step becomes part of the context for the next step. The model effectively **offloads computation into the token stream**. Step 3 can reference the output of step 2 because step 2 is right there in the context. This is why chain-of-thought works — more tokens = more "compute time" = more steps of reasoning the model can chain together.

**How reasoning models are trained differently:**

1. **Training data includes reasoning traces** — Instead of just (question, answer) pairs, the training data contains (question, detailed step-by-step reasoning, answer). The model learns to produce the reasoning steps as part of its output.

2. **Reinforcement learning on reasoning quality** — The model is rewarded not just for correct final answers, but for producing reasoning chains that lead to correct answers. This is often done with outcome-based RL: generate many reasoning traces, check which ones arrive at the correct answer, reinforce those traces.

3. **Test-time compute scaling** — Reasoning models are designed to use more compute *at inference time* (when answering your question) by generating longer chains of thought. Standard models are optimised to be fast and concise. Reasoning models deliberately trade speed for accuracy by "thinking longer."

**The "thinking" you see in the UI:**

When ChatGPT shows a "Thinking..." block that you can expand, those are the intermediate reasoning tokens the model generated before its final answer. Some providers hide these tokens (you pay for them but don't see them). Others show them. The model isn't doing anything architecturally different — it's still predicting the next token. It's just been trained to predict *reasoning-shaped* tokens before *answer-shaped* tokens.

**When to use which:**

| Task type | Better model |
|---|---|
| Quick factual questions, creative writing, simple coding | Standard model (faster, cheaper) |
| Multi-step math, complex logic puzzles, hard coding problems | Reasoning model |
| Tasks where you need to show your work or verify correctness | Reasoning model |
| High-volume, low-latency applications | Standard model |

**The key takeaway:** "Reasoning" models don't have a different architecture or a special reasoning engine bolted on. They're the same transformer, doing the same next-token prediction. The difference is that they've been trained to produce thinking-out-loud tokens that serve as working memory, and this demonstrably improves accuracy on hard problems. It's still token prediction — but token prediction *about the reasoning process itself* rather than jumping straight to the conclusion.

---

### How does training actually happen? Is it unsupervised? What about RLHF?

Training happens in three distinct stages. Each uses a different learning method.

**Stage 1 — Pretraining (self-supervised learning)**

First, what "labels" mean in machine learning: a label is the correct answer you're trying to teach the model. If you're training a model to recognise photos of cats, you show it a photo and the label is "cat" — a human had to write that label by hand. That's supervised learning. Expensive, slow, doesn't scale.

Self-supervised is the trick that makes LLMs possible: **the next word in a sentence IS the label, and it's already there in the text.** No human needs to annotate anything.

Here's exactly how it works, step by step:

1. Take a sentence from the internet: "The capital of France is Paris"
2. Hide the last word. Show the model: "The capital of France is ___"
3. The model guesses. Maybe it says "London" (wrong)
4. Reveal the actual word: "Paris"
5. The model compares its guess to the real answer — that's the label. It was right there in the text all along. No human had to write it.
6. The model adjusts its internal numbers slightly so next time it's more likely to guess "Paris" in this context
7. Move to the next sentence. Repeat.

That's it. Do this for trillions of sentences and the model gradually gets very good at predicting what comes next in any context. Along the way, to get good at this prediction game, it *has* to learn grammar, facts, logic, how code works, how arguments are structured — because all of those things determine what word comes next.

**Under the hood — how the data is actually fed in (vocabulary, batches, masking):**

A few mechanical details that the simple walkthrough above hides:

*The vocabulary* — Before training even starts, you build a **vocabulary**: a fixed list of all the tokens the model knows about. A token is not always a word — it's a sub-word chunk produced by an algorithm called **Byte-Pair Encoding (BPE)** that scans a huge corpus and merges the most frequent character sequences until you have a vocabulary of fixed size (typically 32K–200K tokens). Examples from GPT-4's tokeniser: `"hello"` → 1 token, `"antidisestablishmentarianism"` → 6 tokens, `" the"` (with leading space) is a *different* token from `"the"` (no space), and non-English text often costs more tokens per word. The model's final output layer produces one probability per vocabulary entry — so vocabulary size literally sets the width of the output. The vocabulary never changes after training.

*The (Batch, Context) table — flatten first, then chop into rows.* The entire training corpus is first concatenated into one long token stream, with `<|endoftext|>` inserted at every document boundary by the tokeniser pipeline (nobody adds it by hand — it's automatically appended at the end of every document during preprocessing, and it's a reserved special token defined when the tokeniser is built). For example:

```
[The] [cat] [sat] [on] [mat] [<|endoftext|>] [Dogs] [love] [to] [run] [<|endoftext|>] [Sky] [is] [blue] ...
```

This flat stream is then chopped into fixed-length chunks (the "context length", e.g. 2048 tokens). Each chunk becomes one row of the training tensor:

- **Rows = batch dimension.** Each row is an *independent* training example. The model never mixes information across rows — row 2 has zero visibility into row 1, even if they were adjacent in the flat stream. Batching is purely a GPU-efficiency trick.
- **Columns = context/sequence dimension.** Position within one example.

Because `<|endoftext|>` lands wherever a document happened to end, it can appear *anywhere* in a row — beginning, middle, or end. It's not constrained to any particular column.

*Causal masking — the hard architectural rule.* When predicting the token at column `t`, the model is only allowed to see tokens at columns `0` through `t-1` in the same row. This rule is called **causal masking** (or autoregressive masking) and it's enforced mechanically by the architecture — you cannot look right because right = the future = what you're trying to predict. If you let the model peek at future tokens, you've leaked the answer.

*The `<|endoftext|>` token — a soft, learned reset signal, not a hard wall.* Here's the nuance: causal masking lets the model see *all* tokens to its left in the row, including tokens from a *previous* document that happen to sit to the left of an `<|endoftext|>`. The architecture does not cut the leftward view at `<|endoftext|>`. Example: in the row above, when predicting `"love"`, the model technically sees `[The, cat, sat, on, mat, <|endoftext|>, Dogs]` — doc-1 tokens included. What the model *learns through training* is to heavily discount everything before `<|endoftext|>` because that prior context never helps predict the next token in a new document. So:

- **Hard rule (architecture):** always look at everything to the left in this row.
- **Soft behaviour (learned):** treat `<|endoftext|>` as a wall; attention weights for tokens before it drop close to zero.

*"But doesn't chopping into rows split sentences in half?"* Yes — phrases like `"Dogs love to run"` may straddle a chunk boundary and end up split across two unrelated rows. This is accepted noise. The model isn't trying to memorise specific sentences; it's learning statistical patterns across trillions of tokens. Any common phrase appears thousands of times across the corpus and lands fully within a chunk in the vast majority of those occurrences. The few split copies are irrelevant at scale.

*Training chunking ≠ inference.* The fixed-row chunking is purely a training-time data-preparation step to create uniform GPU batches. At **inference time** (when you're chatting with the model) there's no chunking — your prompt is one continuous context up to the model's context window. The model never "sees" rows or batches when serving you.

**"But won't it just memorise the answers?"**

Good question. Yes, if it only ever saw "The capital of France is Paris" once and you tested it on the exact same sentence — that would be memorising, like solving past exam papers. But here's why that's not what happens:

The model sees the word "Paris" in *thousands* of different contexts: "I flew to Paris last summer", "Paris is known for the Eiffel Tower", "The meeting was held in Paris, France." It doesn't learn "after 'The capital of France is', say Paris." It learns something deeper: that "Paris" is associated with France, with capitals, with Europe, with cities — across many different sentence structures. It's learning the *relationship*, not memorising one specific sentence.

Also: the model has far fewer numbers (parameters) than training examples. GPT-3 has 175 billion parameters but was trained on 300 billion tokens. You can't memorise 300 billion things in 175 billion numbers — it's mathematically forced to *compress*, to find patterns rather than store individual facts. It's like studying for an exam with 1000 questions but you're only allowed one page of notes. You can't write all 1000 answers — you have to understand the underlying principles.

**What are "parameters" exactly?**

Parameters = weights = the numbers inside the model. They are the same thing. When someone says "GPT-4 has 1.8 trillion parameters" they mean: the model is made up of 1.8 trillion individual numbers sitting in a file.

Before training, these numbers are random (meaningless). Training is the process of adjusting them, one tiny nudge at a time, until they collectively encode useful knowledge. After training, those specific number values ARE the model. The model *is* its parameters. There's nothing else.

When you download a model (e.g., Llama 3 from Meta), you're literally downloading a file of numbers. A 70 billion parameter model at 2 bytes per number is ~140 GB. That file of numbers is the entire model — all its knowledge, all its abilities. Run those numbers through the transformer architecture (the code that does the math on those numbers) and you get a working LLM.

So when you see "parameters" in any AI context — it just means the numbers that the model learned during training. More parameters = bigger model = more capacity to store patterns = generally smarter (but also more expensive to run).

So: it's not memorising specific sentences. It's learning *patterns about language and the world* that let it handle sentences it has never seen before — including ones that didn't exist when it was trained.

**Where is the knowledge stored?**

In the **weights** — the billions of numbers that make up the model. That's it. There is no separate database, no hard drive it looks things up in, no file full of facts. Everything the model "knows" is encoded in the specific values of those numbers.

Think of it this way: a trained musician doesn't store songs in a filing cabinet in their brain. The knowledge of music is distributed across their neural connections — the way their fingers move, their sense of rhythm, their feel for harmony. If you asked "where is the music stored?" the answer is: everywhere and nowhere specifically. It's in the *pattern* of connections, not in a single location.

Same with LLMs. The fact "Paris is the capital of France" isn't stored in one specific number. It's spread across thousands of numbers that together encode the relationships between "Paris", "France", "capital", "Europe", and so on. Adjusting any one of those numbers slightly might weaken the model's knowledge of French geography — but no single number *is* that knowledge.

This is why you can't open an LLM and search for a fact like you'd search a database. The knowledge is distributed, compressed, and entangled across billions of numbers. It's also why models sometimes get things slightly wrong — the compression is lossy, like a JPEG image losing some detail to save space.

**"Is it the order of numbers that matters, or their magnitude? And do we know why those specific values?"**

Both matter. Each parameter has a specific position (it sits between specific layers/neurons — that's its role in the architecture) AND a specific value (its magnitude — that's what it learned). Change the position and you've changed what computation it participates in. Change the value and you've changed the result of that computation.

Do we know *why* a specific number ended up at, say, 0.0037 instead of 0.0041? In theory, yes — it's the result of the training process. Billions of training examples each nudged it slightly. It landed at 0.0037 because that value, combined with all the other billions of values around it, minimises prediction errors across the training data. It's deterministic math — not random.

But in practice? No human understands *why* that specific value matters. It's like asking "why is this specific grain of sand in this exact position on a beach?" — technically it's the result of waves, wind, and physics over time. But no one can give you a meaningful human-level explanation for any single grain. The system is just too large.

This is the honest answer: **we know the process that produced the values (training). We can verify they work (the model predicts well). But we cannot give a human-readable explanation for why any individual number is what it is.** It's not mysterious — it's just too complex to narrate, like explaining why every pixel in a photo has its specific RGB value. The process is understood. The individual outcomes are too numerous to explain one by one.

**What you get after pretraining — the "base model":**

Imagine you trained someone purely by having them read the entire internet — every book, every article, every Reddit thread, every codebase. They absorbed all of it. Now you talk to them. What happens?

They don't *answer* you. They *continue* whatever you said as if writing the next paragraph. If you type "What is 2+2?" they might respond with "What is 3+3? What is 4+4?" — because on the internet, questions are often followed by more questions (think quiz pages, FAQ lists). They're not *trying* to be helpful. They're just continuing the pattern.

This is the base model. It knows everything but doesn't know it should answer you. It's like a person who read every book in a library but was never taught how to have a conversation. It's a text-continuation engine — not a chat assistant. OpenAI doesn't ship this to users. It needs more training (stages 2 and 3) before it becomes the ChatGPT you actually talk to.

**This is exactly the distinction you see in Azure AI Foundry (and Hugging Face).** When you browse models, you'll often see two flavours of the same model family:

- `Llama-3-70B` — the **base model**. Pretraining only. It completes text, doesn't follow instructions, doesn't behave like a chat assistant.
- `Llama-3-70B-Instruct` (or `-Chat`) — the same base model after **SFT + RLHF** (stages 2 and 3, below). It answers questions, follows instructions, behaves like ChatGPT.

Not all providers ship the base model. OpenAI, for example, does not give you GPT-4 base — only the post-trained version. Meta (Llama), Mistral, and others do release base models publicly, because researchers want them for fine-tuning on their own data. For 99% of applications you want the `-Instruct` / `-Chat` variant; the base model is only useful if you're doing your own SFT/RLHF.

**Stage 2 — Supervised Fine-Tuning (SFT)**

Human contractors (often thousands of them) write ideal examples of conversations: "Given this user message, here is the ideal helpful response." The model is trained on these (input, output) pairs using standard supervised learning. This teaches the model the *format* of being helpful — answer questions directly, follow instructions, be concise, use appropriate tone.

This is what turns a base model into something that behaves like a chat assistant.

**Stage 3 — RLHF (Reinforcement Learning from Human Feedback)**

The fine-tuned model generates multiple responses to the same prompt. Human raters compare pairs of responses and indicate which one is better (more helpful, more accurate, less harmful). They don't write new responses — they just rank existing ones.

These thousands of pairwise human preferences are used to train a separate **reward model** — a smaller model that learns to predict what a human would prefer. It outputs a score: "this response would get rated 7/10 by a human."

The LLM is then optimised using reinforcement learning (specifically an algorithm called PPO — Proximal Policy Optimisation) to maximise the reward model's score. The LLM generates, the reward model scores, the LLM adjusts to score higher. This loop runs for many iterations.

RLHF is how you steer the model toward being helpful, honest, and safe — qualities that are hard to define in explicit rules but easy for humans to recognise when comparing two options.

**Yes — all three stages are just adjusting the same numbers.**

That's the punchline. There is no other mechanism. All three stages do the same fundamental thing: nudge the model's parameters (numbers) to be slightly better. The difference is *what "better" means* at each stage:

| Stage | What "better" means | What changes |
|---|---|---|
| Pretraining | "Better at predicting the next word" | The same numbers |
| SFT | "Better at responding like a helpful assistant" | The same numbers |
| RLHF | "Better at producing responses humans prefer" | The same numbers |

It's the same file of numbers being refined three times, with three different definitions of "good." Think of it like sculpting: pretraining carves the rough shape (knowledge), SFT refines the form (helpfulness), RLHF polishes the surface (safety, tone, preference). Each stage changes the same block of stone — just with finer tools and a different goal.

After all three stages, you have one file of numbers. That's ChatGPT (or Claude, or Gemini). There is nothing else running behind the scenes — no secret database, no rule engine, no internet connection. Just numbers being multiplied together really fast.

**Can an instruct model be fine-tuned again after RLHF?**

Yes. RLHF does not lock or finish the model; it only produces another checkpoint whose weights can be updated again. Enterprises usually start with an instruct model and apply SFT, often through LoRA/QLoRA adapters, using examples of their desired tasks, terminology, formats, or tool calls. Preference methods such as DPO or RL can also be applied again when preference data is available.

Further training can weaken general instruction-following or safety behaviour, so it normally uses a low learning rate, high-quality data, and regression evaluations. Not every instruct model uses traditional RLHF; some use SFT followed by preference optimisation such as DPO.

---

### If it's all just numbers, is more parameters the only way to make models better?

No. That was the belief from roughly 2020–2023 — "just make it bigger and it gets smarter." And it worked! GPT-2 (1.5B parameters) → GPT-3 (175B) → GPT-4 (rumoured ~1.8 trillion) showed clear jumps in capability with scale. This was called the **scaling laws** era: more parameters + more data + more compute = better model, predictably.

But there are multiple levers, not just size. Here's the full picture:

**The levers you can pull to improve a model:**

1. **More parameters** — Bigger model, more capacity. But: diminishing returns, and cost grows fast (training GPT-4 reportedly cost $100M+).

2. **More/better training data** — A smaller model trained on higher-quality data can beat a bigger model trained on garbage. This is where things shifted recently. Companies now heavily curate and filter training data rather than just scraping everything.

3. **Better architecture** — The Transformer (invented 2017) was a huge leap over previous architectures. Small improvements to the architecture (better attention mechanisms, mixture-of-experts where only part of the model activates per token) give gains without adding raw size.

4. **Better training recipes** — Learning rate schedules, data ordering, curriculum (easy examples first, hard later), longer training on the same data. A lot of recent progress comes from training the *same size* model more carefully.

5. **Post-training improvements** — Better RLHF, better fine-tuning data, techniques like DPO (Direct Preference Optimisation — a simpler alternative to RLHF). This is cheap compared to pretraining and gives big quality bumps.

6. **Test-time compute** (reasoning models) — Instead of making the model bigger, let it *think longer* when answering. The reasoning models (o1, o3) are this: same-ish model, but it generates more tokens (chain of thought) before answering, trading speed for accuracy.

**Where are we hitting walls?**

**Wall 1 — Data.** This is the biggest one. We've nearly run out of high-quality text on the internet. Models have been trained on essentially all of publicly available human writing. You can't just "get more data" anymore. Companies are now buying book licenses, generating synthetic data (using models to generate training data for other models), and exploring video/audio as new data sources.

**Wall 2 — Cost and energy.** Training a frontier model costs $100M–$1B in compute and uses as much electricity as a small city for months. Doubling parameters roughly quadruples the training cost. There's a physical limit to how many GPUs you can wire together and how much power you can supply.

**Wall 3 — Diminishing returns from scale.** Going from 7B to 70B parameters gives a huge intelligence jump. Going from 70B to 700B gives a smaller one. Going from 700B to 7 trillion gives an even smaller one. Each 10x in size gives less and less improvement. We can't just keep scaling forever and expect linear progress.

**Wall 4 — Inference cost.** Even if you could train a 10-trillion-parameter model, running it (inference) would be too slow and expensive for real products. Users expect responses in seconds, not minutes.

**Where progress is actually happening now (2025–2026):**

| Direction | What it means |
|---|---|
| **Smaller, smarter models** | Train smaller models better (better data, longer training, distillation from bigger models). Llama 3 8B rivals GPT-3.5 which is 20x larger. |
| **Reasoning at inference time** | Let models think longer on hard problems. Costs more per query but no retraining needed. |
| **Synthetic data** | Use existing good models to generate training data for new models. Controversial but working. |
| **Mixture of Experts (MoE)** | Model has many "expert" sub-networks but only activates a few per token. Gets the capacity of a huge model at the inference cost of a small one. |
| **Multimodal training** | Train on images, video, audio — not just text. New data sources that haven't been exhausted yet. |
| **Better post-training** | RLHF improvements, DPO, constitutional AI. Get more out of the same base model with better alignment techniques. |
| **Longer context windows** | Let models process more text at once (100K+ tokens). Doesn't make them "smarter" but makes them more useful for real tasks. |

**The short answer to your question:** More parameters was the easy answer from 2020–2023. It still helps, but we've hit diminishing returns and cost walls. Progress now comes from being *smarter* about training (better data, better recipes, better architectures, reasoning at inference time) rather than just brute-forcing scale. The field shifted from "just make it bigger" to "make the most of what we have."

---

### What is backpropagation? Are human brains like this?

**The setup — what is a neural network:** A neural network is layers of mathematical operations connected by **weights** (just numbers, typically billions of them). Input data flows forward through these layers, each layer transforming it, until a final output is produced. This forward flow is called the **forward pass**.

**What backpropagation does:** Backpropagation ("backprop") is the algorithm that lets the network learn from its mistakes. It works in four steps:

1. **Forward pass** — Run an input through the network, get an output (e.g., predicted next token)
2. **Calculate loss** — Compare the output to the correct answer. The difference is called the *loss* (a number representing how wrong the model was)
3. **Backward pass** — Using calculus (specifically the chain rule for derivatives), trace backward through every layer to determine how much each individual weight contributed to the error. This gives you a *gradient* for each weight — a direction and magnitude indicating "if you increase this weight, the error goes up/down by this much"
4. **Update weights** — Nudge each weight slightly in the direction that reduces the error (this step is called *gradient descent*)

Repeat this cycle billions of times across the training data. The weights gradually converge to values that minimise prediction errors.

**Analogy:** Imagine calibrating a machine with a million dials. The machine produces wrong output. Backpropagation tells you exactly which dials contributed most to the error and exactly which direction to turn each one to fix it. Do this millions of times and the machine becomes accurate.

**But how does it actually know which dial is the culprit?**

The honest answer is **calculus** — specifically, the chain rule from high-school calculus, applied automatically to every weight in the network.

The model is a giant composite function: input → layer 1 → layer 2 → ... → layer 96 → output. Each layer multiplies by some weights and applies a simple function. The whole thing is one massive mathematical expression with billions of variables (the weights).

When we get a loss number (say, the model predicted "London" but the answer was "Paris"), we want to know: **for each weight, if I nudged it up by a tiny amount, would the loss go up or down, and by how much?** That "how much loss changes when I change this one weight" is the **gradient** of the loss with respect to that weight. It's a derivative.

The chain rule says: if `y = f(g(h(x)))`, then derivatives multiply backwards through the chain. A neural network is exactly this — a long chain of functions. So the derivative of the loss with respect to a weight in layer 1 = (derivative of loss w.r.t. layer 96 output) × (derivative of layer 96 w.r.t. layer 95) × ... × (derivative of layer 2 w.r.t. layer 1). A long product, multiplied backwards from output to input.

The "backward pass" mechanically:

1. **Forward pass.** Run input through network. Save every intermediate value (the output of every layer).
2. **Compute loss.** Compare final output to correct answer. Get a single number.
3. **Compute gradient at the output.** Easy — the derivative of the loss with respect to the output is a simple formula.
4. **Propagate backward, layer by layer.** At each layer, you already have the gradient flowing in from the layer ahead. Multiply by the local derivative and pass the result back to the previous layer. The chain rule does the bookkeeping.
5. **At each weight, you get a gradient.** It says: "if you increase this weight by ε, the loss will change by approximately (gradient × ε)."

So when you ask "how does it know who is the culprit?" — it doesn't deduce blame in any human sense. The math just falls out. A weight that had a big effect on the wrong output gets a big gradient. A weight that had little effect gets a tiny gradient. It's not "this weight is to blame" — it's "the math says changing this weight by 1 unit would change the loss by 0.0003, while changing that one by 1 unit would change it by 0.07 — so the second one is more responsible, adjust it more."

Then each weight is updated: `new_weight = old_weight - learning_rate × gradient`. The minus sign means "move opposite to the gradient" (the gradient points toward *increasing* loss, and we want to decrease it). The learning rate is a small number (e.g., 0.0001) so we take tiny steps and don't overshoot.

**Modern frameworks (PyTorch, TensorFlow) do all this automatically** via a feature called **autograd**. You write the forward pass; the framework records every operation and automatically computes the backward pass. You never compute derivatives by hand. This is why deep learning became practical — you don't need to be a calculus wizard to train a model.

**Are human brains like this?**

Superficially similar, fundamentally different:

| Aspect | Neural Networks | Human Brains |
|---|---|---|
| Basic unit | Artificial neuron (mathematical function) | Biological neuron (electrochemical cell) |
| Connections | Weights (numbers adjusted by backprop) | Synapses (strengthened by repeated firing — Hebbian learning: "neurons that fire together wire together") |
| Learning signal | Explicit error signal propagated backward | No known backward error signal — brains likely use different local learning rules |
| Data efficiency | Needs billions of examples | Learns from very few examples (a child sees a dog 3 times and generalises) |
| Timing | Trained once, then frozen | Learns continuously throughout life |
| Embodiment | None — processes text/numbers | Fully embodied — tied to senses, emotions, survival instincts, motor control |

The brain inspired the *idea* of neural networks (layers of connected units that learn by adjusting connection strengths), but the specific mechanism of backpropagation has no known biological equivalent. Brains don't do calculus on error gradients. They use local learning rules (each synapse adjusts based on local activity) and are shaped by evolution, not optimisation algorithms.

Use the analogy for intuition. Don't take it literally.

---

### A complete training example, end to end

Let's trace one single training step in detail using everything covered above. Tiny example so the numbers are followable, but the principles scale exactly to GPT-4.

**Setup:**

- Vocabulary: 50,000 tokens
- Model: small transformer, say 100M parameters, 12 blocks
- Training text snippet: `"The capital of France is Paris"`
- After tokenisation: `[The] [ capital] [ of] [ France] [ is] [ Paris]` — 6 tokens with IDs say `[464, 3139, 286, 4881, 318, 6342]`

**Step 1 — Prepare input and target.**

The model is trained to predict the next token at every position. From this one sentence, we create 5 prediction tasks at once:

| Position | Input (sees) | Target (must predict) |
|---|---|---|
| 1 | `The` | ` capital` |
| 2 | `The capital` | ` of` |
| 3 | `The capital of` | ` France` |
| 4 | `The capital of France` | ` is` |
| 5 | `The capital of France is` | ` Paris` |

All 5 predictions happen in **one forward pass** thanks to causal masking — each position only sees tokens to its left.

**Step 2 — Forward pass.**

1. Each token ID is converted to an **embedding** — a learned vector of size, say, 768. So now we have a 6×768 matrix.
2. **Positional information** is added or applied (for example, with RoPE) so the model knows token order.
3. The embeddings pass through **transformer block 1**: attention (tokens look left and gather info), then feed-forward (each token's vector refined).
4. Output of block 1 → block 2 → block 3 → ... → block 12.
5. After the final block, we have a 6×768 matrix.
6. The final **output projection** layer multiplies this by a 768×50,000 matrix, producing a 6×50,000 matrix of **logits** (raw scores, one per vocabulary token).
7. Apply **softmax** (a function that turns logits into probabilities summing to 1). Each row is now a probability distribution over the vocabulary.

So at position 5, after seeing `"The capital of France is"`, the model outputs probabilities for all 50,000 possible next tokens. Maybe:

- `Paris` → 0.42
- `the` → 0.08
- `Lyon` → 0.05
- `London` → 0.03
- ...everything else summing to 0.42

**Step 3 — Compute loss.**

For each of the 5 positions, compare the predicted distribution to the actual answer. The loss function is **cross-entropy loss**. Intuitively: high when the model assigned low probability to the correct token, low when it assigned high probability.

Shape of the formula: `loss = -log(probability assigned to correct token)`.

At position 5, the model gave `Paris` probability 0.42 → loss = `-log(0.42) ≈ 0.87`.

If it had given `Paris` probability 0.99, loss would be `-log(0.99) ≈ 0.01` — almost zero. If it had given 0.001, loss would be `-log(0.001) ≈ 6.9` — large.

Average the loss across all 5 positions → one number, e.g., `1.34`. Total loss for this example.

**Step 4 — Backward pass.**

The framework (PyTorch, etc.) traces backwards through every operation that contributed to the loss. For each of the 100 million weights, it computes a gradient via the chain rule (as described in the backprop section above):

- A weight in the attention of block 7 that affected how `France` influenced position 5: gradient = 0.003
- A weight in the embedding for the token `Paris`: gradient = -0.012
- A weight in some unrelated part of the network: gradient = 0.0000001

Autograd does this automatically. No human writes derivative formulas.

**Step 5 — Update weights.**

For each weight: `new_weight = old_weight - learning_rate × gradient`. With learning rate = 0.0001:

- Weight with gradient 0.003 → new value = old - 0.0000003 (tiny nudge down)
- Weight with gradient -0.012 → new value = old + 0.0000012 (tiny nudge up)

Each weight moves a hair's breadth. Individually meaningless.

**Step 6 — Repeat.** Next training example. Do it all again. And again.

**Scale:** each training batch processes thousands of examples in parallel on GPUs. A full training run processes trillions of tokens. For a frontier model, this takes months on tens of thousands of GPUs. After all that — billions of tiny weight adjustments later — the weights have settled into values that minimise the loss across the entire training corpus. That accumulated competence is what we experience as "the model knows things."

**Key insight:** every interaction you have with ChatGPT is the *inference-time* version of step 2 above (the forward pass). The model isn't learning from your conversation — it's just running its already-trained weights forward to produce probabilities, sampling from them (per the temperature/top-p section), and outputting tokens. All the learning happened during training; inference is frozen.

---

### Why do LLMs hallucinate?

Because their fundamental job is to generate **plausible-sounding text**, not **verified true text**. These are not the same thing, and nothing in the training process guarantees truth.

Five root causes:

**1. The prediction mechanism itself** — The model generates whatever token is statistically most probable given the context. "Most probable" ≠ "factually correct." A plausible-sounding citation in an academic style might be entirely fabricated — because that's what citation-shaped text looks like statistically, even if the specific paper doesn't exist.

**2. No external lookup at inference time** — When you ask a question, the model doesn't search a database or check Wikipedia. It generates purely from compressed knowledge stored in its weights. Compression is lossy (information is lost during compression). Gaps in knowledge get filled with plausible-sounding substitutes rather than "I don't know."

**3. No internal uncertainty signal** — The model produces tokens with the same mechanical confidence whether it's reciting a well-known fact or inventing one. It has no reliable internal "I'm not sure about this" flag. Every token is generated the same way regardless of the model's actual accuracy on that specific claim.

**4. Training data quality** — The internet contains misinformation, contradictions, outdated facts, and fiction presented as fact. The model learned from all of it without a way to distinguish reliable sources from unreliable ones during pretraining.

**5. Context pressure** — If the surrounding conversation strongly implies a certain kind of answer (e.g., the user is clearly expecting a specific name/date/fact), the model is statistically pulled toward generating *something* that fits that slot, even if it has to fabricate it. Saying "I don't know" is less probable in contexts where the user expects a concrete answer.

**How hallucination is reduced in practice:**
- **RAG** — Give the model actual source documents to reference at query time, so it generates from evidence rather than memory
- **Lower temperature** — Temperature controls randomness in token selection. Lower temperature = model picks highest-probability tokens more consistently = less creative but less likely to confabulate
- **Chain-of-thought prompting** — Force the model to reason step by step before answering, which surfaces logical inconsistencies earlier
- **Output validation** — A second model or rule-based system checks the first model's claims against known sources
- **Grounding with structured data** — Connect the model to databases/APIs for factual queries instead of relying on parametric memory (the knowledge stored in weights)

---

### Why Is This Called Emergence?

Pattern learning itself was expected. To predict text, a model benefits from learning grammar, concepts, cause and effect, proof structures, and common reasoning patterns. It does not store every written deduction separately; it learns compressed, distributed regularities that can sometimes be recombined for new problems.

In LLM research, a capability is called **emergent** when it was not explicitly programmed or directly targeted, appears much more successfully in larger models than smaller ones, and was difficult to predict by extrapolating smaller-model results. It does not mean that the capability appeared magically or without a cause.

Researchers expected better next-token prediction as models grew. What was less certain was that the same objective would produce broad abilities such as learning tasks from prompt examples, following new instructions, writing programs, and combining skills across domains without separate training for every task.

The term is debated. Some apparently sudden abilities result from binary evaluation: an almost-correct answer scores `0`, while a completely correct answer scores `1`. When partial progress is measured continuously, some capabilities improve smoothly rather than appearing at one sharp threshold.

For example, if a solution requires five correct steps, gradual per-step improvement can look like a sudden capability jump:

```text
60% reliability per step → 0.6⁵ ≈ 8% complete solutions
90% reliability per step → 0.9⁵ ≈ 59% complete solutions
```

Therefore, a careful interpretation is: scaling gradually improves internal representations and their reliability, while complete-task benchmarks can make the resulting capability look abrupt.

### Why Does Scaling Help?

Scaling means balancing **model parameters, training data, and training compute**, not merely adding parameters.

1. **More representational capacity:** More parameters let the model represent more features and finer distinctions. A smaller model must reuse limited capacity for many patterns, causing them to interfere with one another.
2. **Better composition:** Complex tasks require combining several learned operations, such as understanding a question, selecting a rule, tracking variables, and producing an answer. More layers, dimensions, and attention capacity can make these combinations more reliable.
3. **Better coverage of rare patterns:** More data provides additional examples of uncommon facts, code, proofs, and specialised language. More parameters help retain these patterns without sacrificing common ones.
4. **Better predictive representations:** Correctly continuing text often requires modelling entities, context, time, and cause and effect. Lower prediction loss therefore encourages increasingly useful internal representations of language and the world described by it.
5. **Higher end-to-end reliability:** Small improvements at each step multiply across a multi-step task, producing a much larger improvement in complete solutions.

More parameters alone are not sufficient: a model also needs enough high-quality data and compute to train them. Architecture, training objectives, post-training, and inference-time computation also affect capability. Scaling improves performance empirically, but researchers still do not have a complete mechanistic explanation for every ability it produces.

---

## 7. Glossary

| Term | Definition |
|---|---|
| **LLM** | Large Language Model. A neural network trained on massive text data that can generate, summarise, translate, and reason over language |
| **LCEL** | LangChain Expression Language. The pipe (`\|`) syntax for composing chains declaratively |
| **DAG** | Directed Acyclic Graph. A graph where edges have direction and no cycles exist. LangChain's execution model |
| **Node** | A single computation step in a LangGraph graph, implemented as a Python function |
| **Edge** | A connection between nodes in LangGraph. Can be fixed (always follows) or conditional (decided at runtime) |
| **State** | The shared typed object all nodes in a LangGraph graph read from and write to |
| **Checkpointer** | Persistence mechanism that saves graph state after every node, enabling resumption and replay |
| **Thread ID** | A unique identifier that ties a graph execution to its stored checkpoints |
| **ReAct** | Reasoning + Acting. An agent loop: reason → act → observe → repeat until done |
| **RAG** | Retrieval-Augmented Generation. Fetch relevant documents, insert them into the prompt, generate a grounded response |
| **Embedding** | A list of numbers (vector) representing the meaning of text. Similar meanings → similar numbers |
| **Vector Store** | A database optimised for storing embeddings and finding similar ones fast (e.g., Pinecone, Chroma, FAISS) |
| **Chunking** | Splitting large documents into smaller pieces so each piece gets a focused, precise embedding |
| **Token** | The unit LLMs process text in. Roughly ¾ of a word. You pay per token for API calls |
| **Tool** | A function an LLM agent can invoke at runtime (web search, calculator, API call, database query, etc.) |
| **Human-in-the-Loop** | A workflow pattern where execution pauses for human review or approval before continuing |
| **Serialisation** | Converting an in-memory object into a storable/transmittable format (JSON, bytes) and back |
| **Non-deterministic** | Producing different outputs for the same input across runs. LLMs are non-deterministic by nature |
| **Idempotent** | An operation that produces the same result no matter how many times you run it. Useful for designing retries |
| **Latency** | The time delay between sending a request and getting a response |
| **Self-supervised** | A training method where labels come from the data itself (e.g., predicting the next word). No human annotation needed |
| **Base model** | The raw model after pretraining. Knows a lot but just completes text — not yet an assistant |
| **SFT** | Supervised Fine-Tuning. Training on human-written ideal responses to make the model behave like an assistant |
| **RLHF** | Reinforcement Learning from Human Feedback. Optimising model outputs based on human preference rankings |
| **Reward model** | A model trained on human preferences that scores how "good" a response is. Used to guide RLHF |
| **Backpropagation** | Algorithm for learning: calculate error, trace backward through network to find which weights caused it, adjust them |
| **Gradient descent** | The optimisation step: nudge each weight in the direction that reduces error, proportional to its contribution |
| **Loss** | A number measuring how wrong the model's prediction was. Training minimises this |
| **Weights/Parameters** | The billions of numbers inside a neural network that define its behaviour. Adjusted during training |
| **Emergence** | A capability not explicitly targeted that becomes substantially more successful at scale and was difficult to predict from smaller models; some apparent emergence depends on how performance is measured |
| **Hallucination** | When a model generates plausible-sounding but factually incorrect or fabricated content |
| **Temperature** | Controls randomness in generation. Lower = more deterministic/conservative. Higher = more creative/risky |
| **Diffusion model** | Architecture for image/video generation. Learns to remove noise from random static, guided by text prompts |
| **Modality** | A type of data — text, image, audio, video. Multimodal models handle multiple modalities |
| **Base model** vs **Instruct/Chat model** | Base = pretraining only, completes text. Instruct/Chat = base + SFT + RLHF, behaves like an assistant. Foundry lists both |
| **Vocabulary** | The fixed list of all tokens a model can read or produce. Set during tokenisation, never changes |
| **Tokeniser / BPE** | Byte-Pair Encoding. Algorithm that builds the vocabulary by merging frequent character sequences into tokens |
| **Causal masking** | Rule that a token can only attend to tokens at earlier positions. Prevents the model from cheating by seeing future tokens during training |
| **Attention** | Transformer layer where tokens look at each other and gather relevant information. Uses query/key/value vectors |
| **Feed-forward layer** | Transformer layer applied to each token independently. Refines the info attention gathered. Stores most factual knowledge |
| **Transformer block** | One unit of (attention + feed-forward). Stacked many times (e.g., 96 in GPT-3) to form the full model |
| **RNN / LSTM** | Pre-2017 architecture. Processed text sequentially, one token at a time. Replaced by transformers due to parallelism and long-range issues |
| **Forward pass** | One run of input through the model to produce output. Inference = one forward pass per generated token |
| **KV cache** | Optimisation that caches attention keys/values from earlier tokens so each new token doesn't recompute the full context |
| **Softmax** | Function that converts a vector of raw scores into a probability distribution (positive, sums to 1) |
| **Logits** | The raw, pre-softmax scores the model's output layer produces — one per vocabulary token. Softmax turns them into probabilities |
| **Positional encoding** | Information added or applied to token vectors so attention knows token order and distance. RoPE applies it by rotating query/key vectors |
| **Multi-head attention** | Multiple attention operations run in parallel inside one layer, each learning different types of relationships (syntax, coreference, etc.) |
| **Cross-entropy loss** | The loss function used in language model training. High when the model assigned low probability to the correct token |
| **Gradient** | A number saying how much the loss would change if you nudged a particular weight. Computed by backprop |
| **Learning rate** | The size of the step taken when updating weights. Small (e.g., 0.0001) to avoid overshooting the minimum |
| **Autograd** | Feature in frameworks like PyTorch that automatically computes gradients for any forward pass you define |
| **Sampling** | The process of picking a token from the model's probability distribution. Controlled by temperature, top-k, top-p |
| **Top-k / Top-p sampling** | Strategies that restrict sampling to the most probable tokens (top-k = fixed count, top-p = until cumulative probability ≥ p) |
| **Greedy decoding** | Always pick the highest-probability token. Deterministic but often repetitive |
| **`<\|endoftext\|>`** | Special token marking document boundaries during training. Lets multiple unrelated documents share one row |
