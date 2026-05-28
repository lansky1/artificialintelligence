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

**State** is a typed object (typically a TypedDict or Pydantic model — both are ways of defining a Python dictionary/object where each field has a declared type, so you get autocomplete and error checking) that is shared across all nodes in the graph. Every node receives the full state as input and returns updates to it. This is the critical difference from LangChain chains — there is a single, persistent, structured object that accumulates information as the graph executes. Nodes communicate through state, not just through direct input-output piping.

**Cycles** are what make LangGraph a graph rather than a DAG. A node can route back to a previous node, creating a loop. This is essential for agent behaviour: reason → act → observe → reason again. The loop continues until a termination condition is met (e.g., the agent decides it has a final answer).

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

If you embed an entire 50-page document as one vector, the embedding represents a vague average of the whole document. When a user asks a specific question, this coarse embedding may not match well. Splitting into smaller chunks (e.g., 500–1000 tokens) ensures that each embedding represents a focused piece of information, improving retrieval accuracy.

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

### Follow-up: "But math, code, and medical text were in the training data — how is emergence surprising?"

You're partially right. Math problems, code, and medical textbooks were all in the training data. The model "saw" them. In that sense — not surprising.

The actual surprise is three specific things that go beyond "it saw examples":

**1. Small models trained on the same data can't do it.**

Take a model with 1 billion parameters. Train it on the exact same internet data. It cannot solve novel math problems, reason about hypotheticals, or pass medical exams. Scale that same architecture to 100+ billion parameters with the same data. Suddenly it can.

This discontinuous jump — not gradual improvement, but capabilities appearing abruptly at certain scale thresholds (called "phase transitions") — is what nobody predicted. If it were just "it memorised examples," small models should show proportionally smaller ability. They show essentially *zero* ability until a threshold is crossed. Then they work.

**But why does this happen? Why the sudden jump?**

Honest answer: we don't fully know. But the best current explanation is about **representation depth** — what the model can internally represent at different sizes.

Think of it like this. To solve a math problem, you don't just need to know numbers. You need to simultaneously hold several concepts: what operation is being asked, what the rules of that operation are, how to apply them in sequence, how to track intermediate results. Each of these is a *pattern*. Small models can learn individual patterns (grammar, simple facts) because those are shallow — they don't require combining many things at once. But complex reasoning requires *layering patterns on top of patterns* — using one learned concept as input to another.

A small model is like a person with a tiny desk. They can hold one piece of paper at a time. They can learn individual facts. But a multi-step math problem requires holding 5 pieces of paper simultaneously and combining them. No matter how many times you train them, the desk is too small. It's not a knowledge problem — it's a *capacity* problem.

When the model gets big enough, it has enough parameters to build these layered internal representations: "concept A feeds into concept B which combines with concept C to produce reasoning step D." Below a certain size, these multi-layer representations literally can't fit — the model doesn't have enough numbers to encode them. Above that size, suddenly they can form, and the capability appears all at once.

It's like ice freezing. You cool water gradually: 10°C, 5°C, 2°C, 1°C — still liquid. Then at 0°C, it suddenly becomes ice. The temperature was dropping smoothly, but the *state change* is abrupt. There's a threshold below which the structure can't form, and above which it suddenly can. Same principle with model capabilities.

**2. It generalises to problems it has never seen, not just recalls ones it has.**

There's a fundamental difference between memorising that `∫x² dx = x³/3 + C` and being able to solve a novel integral the model has never encountered in training. A pure lookup system would fail on new problems. LLMs don't — they apply the underlying mathematical structure to solve genuinely new problems. That's generalisation, which is closer to understanding than memorisation.

The baby analogy illustrates this: a child doesn't just repeat sentences it heard. It generates grammatically correct sentences it has *never* heard before by applying internalised rules.

**3. Capabilities compose in ways that weren't explicitly trained.**

The model can take a math problem written in French, solve it, explain the reasoning in English, then write Python code to verify it — in one response. No training example contained that exact combination as a complete task. The model assembles skills from different domains into novel configurations. This compositional generalisation — using learned capabilities together in new ways — is what surprised researchers most.

**The honest synthesis:** You're right that calling it "surprising" can be overstated — if you compress all of human reasoning into a model, some reasoning ability emerging isn't conceptually shocking. The genuine surprise was the *degree* of it, the *discontinuity* with scale (it's not gradual), and the fact that a single training objective — predict the next token — was sufficient to produce all of it without task-specific training. Most researchers expected you'd need specialised training for each capability.

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
| **Emergence** | Capabilities appearing at scale that weren't explicitly trained — e.g., reasoning, arithmetic from next-token prediction |
| **Hallucination** | When a model generates plausible-sounding but factually incorrect or fabricated content |
| **Temperature** | Controls randomness in generation. Lower = more deterministic/conservative. Higher = more creative/risky |
| **Diffusion model** | Architecture for image/video generation. Learns to remove noise from random static, guided by text prompts |
| **Modality** | A type of data — text, image, audio, video. Multimodal models handle multiple modalities |