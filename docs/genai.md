# Generative AI

## 1. LangChain

### The Problem It Solves

Building apps on top of Large Language Models (LLMs) means doing the same low-level work over and over: building prompt strings, calling each provider's API, parsing messy text responses, and wiring up retrieval. Every new project re-implements these same patterns from scratch.

**LangChain** is a framework — a pre-built toolkit of reusable code — that gives you standard building blocks for all of this. It handles the plumbing (the tedious glue code between components) so you can focus on your actual application logic.

### Core Abstractions

**Prompt Templates** are reusable prompts with placeholder variables. You declare a template once and fill it with different inputs at runtime, instead of hand-building strings with f-strings or concatenation. This keeps prompt wording separate from your code.

**LLM Wrappers** give you one interface across all model providers (OpenAI, Azure OpenAI, Google Gemini, Anthropic, open-source models via Ollama, and more). Swap providers by changing one line of config — no rewrite needed.

**Chains** are the basic way to compose steps. A chain feeds the output of one step into the next: format a prompt → call the LLM → parse the output. Each step transforms the data and passes it forward.

**LCEL (LangChain Expression Language)** is the modern syntax for chains, using the `|` (pipe) operator. It replaces older, verbose chain classes with a short, readable pipeline:

```python
chain = prompt_template | llm | output_parser
result = chain.invoke({"topic": "quantum computing"})
```

LCEL chains are **lazy** — they describe the steps but don't run them yet. Nothing executes until you call one of:

- `.invoke()` — run once
- `.stream()` — get results token-by-token as they're generated
- `.batch()` — run the same chain on many inputs in parallel

They also support async execution (non-blocking): your program can do other work while it waits for the LLM, instead of freezing.

**Output Parsers** turn the LLM's raw text into structured data. By default an LLM returns just a string of words, so parsers extract JSON, lists, Pydantic objects, or other formats. (**Pydantic** is a Python library for defining data shapes with type checks — e.g. forcing `name` to be a string and `age` to be an integer.) Parsers can also inject formatting instructions into the prompt automatically.

**Retrievers** connect the LLM to outside knowledge. Instead of relying only on training data, a retriever fetches relevant documents from a vector store, database, or search engine and adds them to the prompt. This is the basis of RAG (covered below).

**Agents** let the LLM decide which tools to call and in what order. Rather than following a fixed chain, the LLM reasons about the next action, runs it, observes the result, and chooses what to do next. Early LangChain agents worked but were brittle — they lacked structured state and reliable looping, which is why LangGraph was created.

### The Architectural Limitation

LangChain chains form a **DAG (Directed Acyclic Graph)** — a graph where connections only go forward, like a one-way street system with no U-turns.

- **Directed** means edges have a direction (A→B, not just A—B).
- **Acyclic** means no loops — you can never follow the edges and end up back where you started.

Data flows forward through the steps and can branch into parallel paths, but it never loops back. So a DAG has no built-in way to:

- Re-run a step based on its own output
- Keep shared, changeable state that many steps read and write
- Branch conditionally onto entirely different paths

This makes LangChain great for linear pipelines but not enough for complex agent workflows that need iteration, decisions, and memory.

### Is a Diamond Pattern Valid in a DAG?

Yes. "Acyclic" only bans loops. Branching out and merging back is fine, because data still only flows forward.

```
        ┌──── branch A ────┐
input ──┤                  ├──── output parser ──── final output
        └──── branch B ────┘
```

LCEL gives you two ways to express this.

**`RunnableParallel`** runs both branches at the same time and merges their outputs into a dict:

```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    path_a=chain_a,
    path_b=chain_b
)
# output is {"path_a": ..., "path_b": ...}
diamond_chain = parallel | merge_and_parse
```

**`RunnableBranch`** picks exactly one branch based on a condition (the others don't run):

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: x["type"] == "A", chain_a),
    chain_b  # default
)
diamond_chain = branch | output_parser
```

So: `RunnableParallel` runs all branches and merges; `RunnableBranch` routes to one. Both are valid DAGs. What breaks the DAG model is a loop — a node's output feeding back into an earlier node. That is exactly what LangGraph adds.

### Can't I Just Use RunnableBranch for Tool Fallbacks?

Yes — for a **fixed, known number of fallbacks**. This is a valid DAG:

`tool_a → evaluate → RunnableBranch(sufficient → output_parser, insufficient → tool_b → output_parser)`

It breaks down when you don't know the number of tries upfront. If `tool_b` also fails, you'd nest another branch inside, then another inside that. A DAG must be **fully defined before it runs** — every node and edge has to exist before you call `.invoke()`. You can't encode "keep trying until it works" in a static graph.

LangGraph solves this because the routing decision (loop again or exit) is made at *runtime* by a conditional edge that inspects the current state. The graph never needs to know in advance how many iterations it will take.

---

## 2. LangGraph

### The Problem It Solves

Many real-world LLM applications are not simple pipelines. An AI agent might need to call a tool, judge the result, decide it is not good enough, call a different tool, judge again, and repeat until satisfied. A multi-agent system might need several agents collaborating on a shared workspace. A workflow might need to pause for human review before continuing.

None of these fit cleanly into a DAG (a one-way graph with no loops). **LangGraph** is a framework, built on top of LangChain, that models LLM applications as stateful, cyclical graphs. Loops, conditional routing, shared state, and persistence are all built in.

### Core Concepts

Picture a team of workers connected by arrows, all writing on a shared whiteboard.

- **Nodes** are the workers — the units of computation. Each node is a Python function that reads the current state, does some work (call an LLM, run a tool, transform data), and returns an updated state. Your actual logic lives here.

- **Edges** are the arrows — they decide which node runs next. There are two kinds:
  - *Fixed edges* — unconditional. After node A, always go to node B.
  - *Conditional edges* — a routing function looks at the current state and picks the next node.

- **State** is the whiteboard — a typed object (usually a TypedDict or Pydantic model, both ways of defining an object where each field has a declared type, giving you autocomplete and error checking) shared by every node. A node reads a snapshot of the state and returns only its own updates. LangGraph then merges those updates field by field using **reducers** (rules for combining a node's update with the existing value — for example, append to a list rather than overwrite it). This is not a freely mutable global dictionary; it is runtime-managed state that accumulates as the graph runs.

- **Cycles** are what make this a graph rather than a DAG. A node can route back to an earlier node, forming a loop. This is essential for agents: reason, act, observe, reason again. The loop continues until a stopping condition is met (for example, the agent decides it has a final answer).

### Why Didn't LangChain Use Global State?

It could have. Apps already used memory, agent scratchpads, or dictionaries passed through chains. LCEL deliberately favoured explicit `input → output` steps so each call stays isolated and its data dependencies stay visible.

A freely mutable global object causes trouble when you batch, run asynchronously, or run parallel branches. Imagine many people scribbling on one whiteboard at once: they overwrite each other's notes, read half-finished updates, or need locks that kill the parallelism. It also makes streaming harder, because raw mutation says nothing about chunk ordering, completion, backpressure, or which call owns an update.

LangGraph keeps the useful version of shared state. Parallel nodes read the same snapshot and return partial updates, which the runtime merges at a controlled point using declared reducers. So shared state itself was never the problem — hidden, uncontrolled shared mutation was.

### Why Not Just Use a Python Loop Around a Chain?

You could wrap a LangChain chain in a `while` loop and get iteration. But it falls apart fast:

- **Manual state management** — you must track by hand what carries over between iterations, handle edge cases, and keep things consistent.
- **No conditional routing** — branching logic gets buried in tangled if/else blocks.
- **No observability** — no built-in way to inspect intermediate states, trace execution, or debug a specific step.
- **No persistence** — if the process crashes, all intermediate work is lost.
- **No human-in-the-loop** — no natural place to pause and resume.
- **Doesn't scale to multi-agent** — coordinating several agents with shared state in raw Python becomes unmaintainable.

LangGraph gives you all of this as infrastructure. The graph is declarative (you describe *what* the workflow looks like — which nodes exist and how they connect — instead of writing step-by-step control flow), observable, and can be interrupted and resumed at any point.

### The ReAct Pattern

**ReAct** (Reasoning + Acting) is the dominant pattern for LLM agents. The cycle:

1. **Reason** — the LLM looks at the current state and decides what to do next.
2. **Act** — the chosen action runs (call a tool, query a database, etc.).
3. **Observe** — the result is added to the state.
4. **Repeat** — the LLM re-reads the updated state and either acts again or produces a final answer.

This is a loop, and LangGraph's cyclical graph maps onto it directly. The LLM node routes to an action node, which routes back to the LLM node, until the LLM routes to the "end" node.

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

The two are complementary, not competing. LangGraph orchestrates the overall flow — deciding what runs when, managing state, handling loops — while LangChain components (prompt templates, LLM wrappers, retrievers, output parsers) do the actual work inside individual nodes.

---

## 3. Checkpointers (State Persistence in LangGraph)

### What a Checkpointer Does

A **checkpointer** is a save system for your graph. After every node runs, it automatically saves the graph's state.

To save the state, it first **serialises** it — turns the in-memory Python object into a storable format like JSON or bytes. It then writes that to a backend (memory, SQLite, PostgreSQL, etc.), filed under a thread identifier so it can be found again later.

### Why Persistence Matters

- **Resumability** — If the process crashes, the server restarts, or the user closes their browser, the graph picks up from the last completed step instead of starting over. This matters for long-running workflows.
- **Conversation memory** — In a chatbot, the checkpointer holds the conversation history. When the same user sends a new message (using the same `thread_id`), the graph reloads the full prior state and continues naturally.
- **Human-in-the-loop** — A graph can pause at a chosen node, for example before a sensitive action. It saves the state, waits for human approval (which might take minutes, hours, or days), then resumes exactly where it stopped.
- **Time travel and debugging** — Because state is saved at every step, you can replay from any checkpoint. Inspect the state at the moment something broke, or re-run from that point with different inputs.

### How It Works in Practice

Every graph invocation needs a `thread_id` in its config:

```python
config = {"configurable": {"thread_id": "user-123-session-1"}}
result = graph.invoke(input, config)
```

The checkpointer saves a snapshot after each node finishes, tied to this `thread_id`. Call the graph again with the same `thread_id`, and it loads the latest snapshot and continues from there instead of starting fresh.

### Available Checkpointer Implementations

| Checkpointer | Use Case |
|---|---|
| `MemorySaver` | Development and testing. State lives in-memory and is lost on process restart. |
| `SqliteSaver` | Local persistence for single-user applications or prototyping. |
| `PostgresSaver` | Production deployments with concurrent users, durability requirements, and horizontal scaling. |

---

## 4. Retrieval-Augmented Generation (RAG)

### The Problem It Solves

An LLM only knows what was in its training data, frozen at a **knowledge cutoff** (the date training stopped). It can't see your private documents, real-time information, or domain-specific knowledge that wasn't in that data.

RAG fixes this. At query time, it retrieves relevant external text and pastes it into the prompt, so the model answers from fresh, specific facts instead of memory alone.

### How RAG Works

RAG runs in two phases.

**Indexing** happens offline, once or on a schedule:

1. Load documents from your sources (files, databases, APIs, web pages).
2. Split them into smaller **chunks**. LLMs have a limit on how much text they can read at once, and smaller chunks also retrieve more accurately.
3. Turn each chunk into an **embedding** — a list of numbers (e.g. 1536 floats) that captures the *meaning* of the text. Text with similar meaning gets similar numbers, so you can find related content by comparing the number lists mathematically.
4. Store the chunks and their vectors in a **vector store** — a database built for fast similarity search over these number lists (e.g. Pinecone, Chroma, FAISS, Weaviate).

**Retrieval and generation** happens online, every time someone asks:

1. The user asks a question.
2. The question is embedded with the same embedding model.
3. A **similarity search** finds the closest chunks. Closeness is usually measured with **cosine similarity** (the angle between two vectors — smaller angle means more similar meaning).
4. Those chunks are pasted into the prompt next to the question.
5. The LLM answers, grounded in the retrieved text.

### Why Chunking Matters

Imagine embedding a whole 50-page document as one vector. That single vector has to stand for dozens of unrelated topics at once, so any one useful sentence gets drowned out by everything else. This is **semantic dilution**: the vector becomes a vague average and stops matching specific questions closely.

Smaller chunks keep each embedding focused on one idea, which sharpens retrieval.

**Who creates the chunks?** You pick a strategy and settings; a text-splitting library does the actual cutting. Common strategies are fixed-size, sentence- or paragraph-aware, semantic, and structure-aware splitting. There's no universal best size — you find it by testing retrieval with real questions.

Small and large chunks trade off against each other:

| Small chunks | Large chunks |
|---|---|
| Match focused questions precisely | Preserve more surrounding context |
| Add less irrelevant text to the prompt | Keep related sentences, headings, and definitions together |
| May separate a statement from its heading, subject, or definition | Cause semantic dilution by mixing several topics into one vector |
| May require several chunks to answer one question | Consume more of the LLM's context window |

**Overlap** (repeating a few tokens between neighbouring chunks) keeps information from being cut in half at a boundary. Too much overlap just creates duplicates. A good starting point is 300–800 tokens with 10–20% overlap, then evaluate.

### Parent-Child Retrieval

You can get the best of both: search tiny chunks for precision, but hand the LLM a bigger section for context.

A parent section like `Annual Leave Policy` might hold these child chunks:

```text
Child 1: Employees receive 20 vacation days.
Child 2: Contractors do not receive paid vacation.
Child 3: Unused vacation expires at year-end.
```

Ask `What happens to unused vacation?` and the search lands exactly on Child 3. Each child stores a `parent_id`, so the system then pulls up the whole `Annual Leave Policy` to give the LLM the full context:

```text
search small child → find exact match → return larger parent
```

### Handling Large Documents

Don't chop a big document blindly every N tokens. Treat it as a hierarchy:

```text
document → headings/sections → paragraphs or child chunks
```

First pull out structure — headings, paragraphs, tables, page numbers. Then:

- Use sections as **parent** chunks.
- Split oversized sections into smaller searchable **children**.
- Store metadata on each child, such as `document_id`, `parent_id`, heading, and page.
- Search the children, and fetch the parent when you need more context.

Large collections of documents use the same approach, plus batch ingestion, deduplication, and metadata filters (department, document type, tenant, year) to shrink the search space.

### How an Embedding Model Creates a Vector

```text
text → tokens → token IDs → token vectors → transformer → pooling → text vector
```

1. **Tokenisation:** The tokenizer splits text into vocabulary entries. It might split `Employees receive vacation` into `Employ`, `ees`, ` receive`, and ` vacation`.
2. **Token IDs:** Each entry has an arbitrary integer ID, such as `[8321, 291, 4256, 10982]` (illustrative). These are just lookup keys, not meanings: `8321` simply names one row in the vocabulary.
3. **Token vectors:** Each ID selects a learned row from the model's **embedding matrix**. With a 50,000-token vocabulary and 768-dimensional internal vectors, this matrix has shape `50,000 × 768`.
4. **Contextualisation:** Transformer layers use self-attention to update each token vector based on its neighbours. This is how `bank` can mean different things in financial versus river contexts.
5. **Pooling and normalisation:** Pooling combines all the contextual token vectors into one vector for the whole chunk. With **mean pooling**, each position is averaged on its own:

```text
Employees → [0.2, 0.4, 0.6]
receive   → [0.4, 0.1, 0.3]
vacation  → [0.6, 0.4, 0.0]

mean      → [(0.2+0.4+0.6)/3, (0.4+0.1+0.4)/3, (0.6+0.3+0.0)/3]
          → [0.4, 0.3, 0.3]
```

Real vectors have hundreds or thousands of positions, but the same averaging is applied to every position. Some models instead use a special token such as `[CLS]`, whose vector gathers information from all tokens through attention, or use learned weights so important tokens count for more. The model or API usually pools internally and returns only the final text vector, often normalised for similarity comparison.

Embedding models are trained on pairs of related and unrelated text. Training pulls a query closer to its matching passage and pushes it away from irrelevant ones. Meaning is therefore spread across the whole vector — a single dimension usually has no human-readable meaning on its own.

**What is stored?** A vector-store record usually holds an `id`, the embedding vector, the original chunk, and metadata (document, page, section). Alternatively the store keeps only `id + vector + metadata`, and the chunk text lives in a separate document or key-value store. Either way the link is `vector ↔ chunk ID ↔ text`: similarity search finds the vector, but the matching text is what gets sent to the LLM.

### Why RAG Over Fine-Tuning

| Concern | RAG | Fine-Tuning |
|---|---|---|
| Knowledge updates | Instant — update the vector store | Requires retraining |
| Cost | Low — no GPU training needed | High — training compute costs |
| Hallucination control | High — model has source text to quote | Lower — model may still confabulate ("confabulate" = confidently generate false information that sounds plausible) |
| Transparency | Can show source documents to user | Opaque — no attribution possible |
| Domain adaptation | Limited to what's in retrieved context | Can deeply reshape model behaviour |

Use RAG when you need answers over a specific set of documents. Use fine-tuning when you need to change the model's underlying behaviour, tone, or reasoning patterns.

---

## 5. Observability — LangSmith vs LangFuse

### Why Observability Matters

**Observability** means being able to see what your app actually did at each step. With LLMs, you need it badly for two reasons.

First, LLMs are **non-deterministic**: the same input can give different outputs. A normal function is predictable — `add(2,3)` always returns 5 — but a model is not.

Second, chains and agents run in multiple steps, and any step can fail or go off-script. Without tooling, debugging is nearly impossible. You can't see what prompt was actually sent, what the model returned at each step, how long each took, or where it broke.

### What Both Tools Provide

- **Tracing** — A step-by-step record of a chain or graph run: inputs, outputs, latency (time per step), token usage (a **token** is the unit LLMs count — roughly ¾ of a word, and you pay per token), and errors.
- **Prompt Management** — Version control for prompts: track changes, roll back, A/B test (send some traffic to prompt A and some to B, then compare quality), and iterate without touching code.
- **Evaluation** — Run a test dataset through your pipeline and score the results, either with another LLM as judge or with human reviewers.
- **Monitoring** — Track cost, latency, error rates, and quality over time in production.

### LangSmith

Built by the LangChain team, so it has the tightest integration with LangChain and LangGraph — traces are captured automatically with almost no setup. The UI for exploring traces, managing datasets, and running evaluations is polished.

**Trade-off:** it is cloud-only. All trace data — your prompts, user inputs, and model outputs — goes to LangChain's servers. There is no self-hosted option.

### LangFuse

An open-source observability platform that is **framework-agnostic** (not tied to any one framework). It works with LangChain, LlamaIndex, raw OpenAI calls, or anything else through its **SDK** (a library you import to talk to a service). It covers the same ground as LangSmith — tracing, evaluation, prompt management.

**Key advantage:** it is fully self-hostable — you run it on your own servers instead of someone else's cloud. That keeps sensitive data (prompts, user inputs, model responses) inside your network. This is often mandatory in regulated industries such as finance, healthcare, government, and legal, because of **data residency laws** (rules requiring data to stay within a given country or network).

### Decision Guide

| Consideration | Choose LangSmith | Choose LangFuse |
|---|---|---|
| Data residency requirements | No strict requirements | Must keep data on-premises |
| Framework | Exclusively LangChain/LangGraph | Mixed or non-LangChain stack |
| Setup preference | Managed cloud, minimal ops | Willing to self-host |
| Budget | Free tier sufficient or paid plan acceptable | Need unlimited usage (self-hosted = free) |

Evaluation is covered in depth in the next section (Section 6).

---

## 6. Evaluation

### Why Evaluating LLMs Is Hard

In normal software, a test asserts an exact output: `add(2, 3)` must return `5`. Anything else fails. LLM applications break this model in three ways:

- **Outputs are open-ended.** A good summary can be written a hundred different ways. There is often no single correct answer to assert against.
- **Outputs are non-deterministic** — the same input can produce different text each run (see Section 5). A test that demands an exact string match will fail for a perfectly good answer.
- **Fluent is not correct.** The model is trained to sound plausible, not to be right. A confident, well-written answer can be completely wrong (a hallucination).

So you cannot just check `output == expected`. You need to measure *quality* — and quality is fuzzy. That is what evaluation tackles.

### The Core Idea: A Golden Dataset

Build an **evaluation set** (also called a **golden dataset**): a collection of representative inputs paired with either expected answers or grading criteria. Then score your system against it and re-run it like a test suite whenever you change something.

A few rows might look like this:

```text
input                          expected / criteria
-----------------------------  -----------------------------------------
"What is our refund window?"   "30 days"  (exact)
"Summarise this ticket"        criteria: mentions the customer's issue
"Is this email spam?"          "spam"     (label)
```

Start small — 20 to 50 hand-picked cases that cover your common queries and your known edge cases are far more useful than nothing. Grow the set over time, especially by adding any real failure you find in production.

### Offline vs Online Evaluation

There are two moments to evaluate, and you want both.

- **Offline evaluation** runs *before you ship*, against your golden dataset. It is your test suite: fast, repeatable, run on every change.
- **Online evaluation** runs *in production*, on real user traffic. It catches what your dataset missed — real questions are messier than the ones you imagined.

Offline tells you whether quality held up against your known test cases. Online tells you whether it actually works for real users.

### Ways to Score an Output

There is no single best scoring method. Each fits a different kind of output, so most real systems combine several.

Before choosing a metric, define the **task** and the **cost of each failure**. Do not default to whichever number is easiest to put on a dashboard. The metric becomes the behaviour the team optimises, so it must reward what the product actually needs.

For example, suppose 97% of transactions are legitimate. A useless fraud detector that predicts "legitimate" every time still gets 97% accuracy while catching no fraud at all. This is **class imbalance**: one label is so common that overall accuracy hides failure on the rare, important label. In that case:

- prioritise **recall** when missing a real positive is most costly (for example, fraud or disease),
- prioritise **precision** when false alarms are most costly (for example, blocking a legitimate payment),
- use **F1** when both matter and you need one comparison score,
- report the separate precision and recall values too, because the same F1 score can hide different trade-offs.

**1. Rule-based / exact-match checks**

Compare the output to a known answer with code: string equality, a regex, "is this valid JSON?", "does it contain this value?".

- *Great for* structured output (JSON, a number, a date) and classification, where there genuinely is one right answer.
- *Useless for* free text — there is no single correct sentence to match against.

For classification (sorting inputs into labels), four standard metrics summarise quality. Plainly:

- **Accuracy** — what fraction of all predictions were correct. Most informative when classes are reasonably balanced and mistakes have similar costs.
- **Precision** — of the items you *flagged* as positive, how many really were.
- **Recall** — of the items that really *are* positive, how many you caught.
- **F1** — a single score balancing precision and recall (their harmonic mean, which stays low unless *both* are high — so you can't game it by maxing out just one).

**2. Reference-based metrics (BLEU, ROUGE, METEOR)**

These score how much the output's words and phrases overlap with a reference answer. Roughly: **BLEU** asks how much of the *output* appears in the reference (precision-leaning, from machine translation); **ROUGE** asks how much of the *reference* appears in the output (recall-leaning, from summarisation).

**METEOR** was designed for translation and is more forgiving than BLEU. It aligns individual words using exact matches, word stems, and synonyms, then penalises matches that appear in a fragmented or incorrect order. This means a valid translation using words such as "blocked" instead of "suspended" can receive credit even when its exact phrases differ from the reference.

- *Useful* as a cheap, automatic signal when you have reference texts.
- *Weakness:* even METEOR only approximates meaning. These metrics remain dependent on the chosen reference and can miss valid paraphrases or factual errors.

**3. Perplexity**

**Perplexity** measures how well a language model predicts a held-out text sequence. Lower perplexity means the model assigned higher probability to the actual next tokens, so it is less "surprised" by the text. It is useful when comparing language models or checkpoints on the **same dataset with the same tokenisation**.

Perplexity can expose poor language modelling and incoherent text, but it does **not** measure whether an answer is correct, grounded, safe, or useful. A fluent hallucination can have low perplexity. Scores from models with different tokenisers are also not directly comparable, and many hosted APIs do not expose the token probabilities needed to calculate it.

**4. LLM-as-a-judge**

Use a second capable LLM to grade the output against a written rubric — for example, *"Is this answer faithful to the source? Score 1-5."*

It is popular because it **scales** (cheap and instant compared to people) and it **handles free text** (it judges meaning, not word overlap), which is exactly where rule-based and overlap metrics fail.

The pitfalls are real, so treat the judge's scores as estimates, not truth:

- It can be **biased** — tending to favour longer, more verbose answers, or answers written in its own style.
- It needs a **clear rubric**. Vague instructions give noisy, inconsistent scores.
- It must be **spot-checked by humans**. Periodically grade a sample yourself and confirm the judge agrees.

**5. Human evaluation**

People read the outputs and rate them. This is the **gold standard** for quality — but it is slow and costly, so you cannot run it on every change. Its most valuable job is to *validate the automated judges*: if humans and the LLM judge agree on a sample, you can trust the judge to scale.

| Method | Best for | Main weakness |
|---|---|---|
| Rule-based / exact-match | Structured output, classification | Useless for free text |
| BLEU / ROUGE | Cheap auto-score vs a reference | Rewards surface overlap rather than meaning |
| METEOR | Translation where stems and synonyms should count | Still reference-dependent; only approximates semantics |
| Perplexity | Comparing language models on the same text and tokenisation | Measures predictability, not correctness or task success |
| LLM-as-a-judge | Free text at scale | Biased; needs a rubric and spot-checks |
| Human evaluation | The final word on quality | Slow and expensive |

### Task-Specific Evaluation: Question Answering

First decide what kind of question-answering system you built, because the scoring method changes:

- **Extractive QA** returns a span copied from a source document. Use **exact match** when every word matters, plus **token-level F1** to give partial credit when most of the reference span was recovered. Exact match is deliberately strict; token-level F1 shows whether a zero was actually a near miss.
- **Generative QA** composes a new answer, often from several sources. Exact string matching is too brittle, so grade semantic correctness and answer relevance with a rubric, an LLM judge, or human review. If retrieval is involved, also evaluate the RAG-specific dimensions below.

### Task-Specific Evaluation: RAG

Generic scores are not enough for a RAG pipeline (Section 4), because a RAG answer can fail at two different stages — retrieval or generation. Three questions pin down where:

1. **Context relevance** — did *retrieval* fetch the right chunks? If the relevant text never made it into the prompt, the answer was doomed before generation started.
2. **Faithfulness / groundedness** — does the answer stick to the retrieved text instead of inventing things? An answer can be fluent and wrong if the model ignored the context and confabulated.
3. **Answer relevance** — does it actually answer the question the user asked, rather than drifting onto a related-but-different point?

Splitting quality this way is diagnostic: low context relevance points at your chunking or retriever; low faithfulness points at your prompt or model. The **RAGAS** framework automates exactly these RAG metrics.

### Agent Evaluation

For agents (Section 2), the final answer is not enough — you judge the **trajectory**, the path the agent took to get there. Did it:

- choose the **right tools** for the job?
- call them in a **sensible order**?
- reach the answer **without wasted steps** (no needless loops or redundant calls)?

Two agents can return the same answer while one was efficient and the other burned ten extra tool calls and twice the cost. Evaluating the trajectory catches that.

### What to Measure in Production

Online, track more than quality. Watch the operational numbers too:

- **Cost** — tokens used and dollars spent per request (a **token** is a chunk of text — about ¾ of a word, so ~100 tokens ≈ 75 words — and you pay per token).
- **Latency** — how long users wait for a response.
- **Error rates** — failed calls, timeouts, malformed output.

Pair these with **user-feedback signals**, which are real quality data for free:

- thumbs up / down on a response,
- **edits** — the user rewrote your answer, so it was not good enough,
- **regenerations** — the user asked again, so the first try missed.

### Regression Testing: CI for Prompts

Prompts and model versions are part of your system, so put them **under test** like code. When you change a prompt or upgrade the model, re-run the whole eval set and compare scores against the previous version. Ship only if nothing got worse.

This is **regression testing** — confirming a change did not break something that used to work. It matters more here than in normal software because LLM behaviour shifts in surprising ways: a prompt tweak that helps one case can quietly wreck five others, and a "better" new model can regress on your specific task. Think of it as **CI (continuous integration) for prompts**: an automated gate that re-runs the evals on every change.

### Tools That Run These Evals

You do not build this from scratch. The observability platforms from Section 5 — **LangSmith** and **LangFuse** — manage datasets, run evaluations (including LLM-as-a-judge), and track production metrics. Other names worth knowing:

- **RAGAS** — purpose-built for the RAG metrics above.
- **DeepEval** — a test-suite-style framework for LLM evals (think pytest for prompts).
- **promptfoo** — compares prompts and models side by side against your test cases.

### Decision Guide

| What you are building | What to evaluate / which method |
|---|---|
| Balanced classifier | Accuracy, plus per-class precision and recall |
| Imbalanced classifier (fraud, rare events) | Precision, recall, and F1; do not rely on accuracy alone |
| Structured data extractor | Schema/rule checks + exact match |
| Extractive question-answering | Exact match + token-level F1 |
| Summariser | LLM-as-a-judge on a rubric; ROUGE as a cheap coverage signal |
| Translator | Human/LLM rubric; BLEU and METEOR as automatic signals |
| Language-model checkpoint | Perplexity on a fixed held-out dataset with consistent tokenisation |
| RAG question-answering | Context relevance, faithfulness, answer relevance (RAGAS) |
| Tool-using agent | Trajectory: tool choice, order, wasted steps |
| Anything in production | Quality + cost, latency, error rates + thumbs/edits/regenerations |
| Any prompt or model change | Re-run the golden dataset (regression / CI for prompts) |

---

## 7. How LLMs Actually Work (Foundations, Training & Limitations)

### What else falls under Generative AI besides LLMs?

**Generative AI** is any model that creates new content. LLMs do this for text, but text is just one type of output. Here are the others:

- **Image generation** — Stable Diffusion, DALL-E, Midjourney. These use **diffusion**: starting from random static, the model slowly removes noise until a clear image appears, guided by a text description. A completely different architecture from LLMs.
- **Video generation** — Sora, Runway. Produce video frames from text or images, using diffusion or transformers adapted to handle time.
- **Audio/Music generation** — Suno, MusicLM. Turn text prompts into sound waveforms or musical sequences.
- **Speech synthesis (TTS)** — ElevenLabs, Azure Neural TTS. Turn written text into realistic human speech.
- **Multimodal models** — GPT-4o, Gemini. Handle text, image, and audio in one model. They overlap with LLMs but go further: they can both understand and generate across all these types.
- **Protein/molecular generation** — AlphaFold. Builds 3D molecular structures from amino acid sequences. GenAI applied to biology, not language.

The common thread: every one of these learned statistical patterns from huge amounts of data, then generates new examples that follow those patterns. LLMs do it for language; the rest do it for images, sound, video, or molecules.

---

### Why do we say LLMs "think"?

The honest answer first: they probably don't think the way humans do. Calling it "thinking" is **anthropomorphisation** — giving human qualities to non-human things. But here is what makes it *look* like thinking, and why the line is blurrier than you'd expect.

**What is actually happening.** The model predicts the next token, one at a time, based on everything before it. That's the whole mechanism. There is no separate "reasoning module" or "understanding engine" inside.

**Why that looks like thinking.** When you train on nearly all human writing — books, papers, logical arguments, code, proofs, debates, stories — you don't just learn which word follows which. You learn the *structure of how humans reason*. Every cause and effect, every logical step ever written down, gets compressed into the model's **weights** (the numbers that define its behaviour).

**The baby analogy.** A baby doesn't memorise sentences and replay them. It watches patterns — objects, actions, consequences, grammar — across thousands of interactions. Eventually it starts producing correct sentences it has never heard. Something emerges beyond pure copying. LLMs go through a similar process at a far larger scale: they observe patterns across trillions of tokens and build internal representations that generalise past the exact examples they saw.

Here is the proof this is real learning, not copying. A baby's first words ("mama", "dada") are imitated. But by age 2 to 3, children say things no one ever said to them, like "I goed to the park" or "two mouses." No adult says "goed" or "mouses." The child has *inferred a rule* (past tense adds "-ed", plurals add "-s") and applied it where it doesn't belong. This is called **overgeneralisation**, and it proves the child learned an abstract pattern rather than memorising phrases.

Same with novel sentences. A 3-year-old who heard "the dog chased the cat" might say "the cat chased the butterfly" — a grammatical sentence built from known words in a new arrangement no one taught. The creativity isn't in the words (those are learned); it's in the *novel combinations* assembled from internal rules. LLMs do exactly this: they internalise patterns and generate new combinations that weren't in the training data.

**The key idea — emergence.** At enough scale (billions of parameters, trillions of training tokens), abilities appear that were never explicitly trained. LLMs do arithmetic, write working code, reason about hypotheticals, and pass medical licensing exams — yet none of these was the goal. The only goal was "predict the next token." The reasoning-like behaviour **emerged** from compressing that much structured human knowledge.

**Is it thinking?** We genuinely don't know. We know that compressing all that human reasoning produces something that *behaves* like thinking across many tasks. Whether there is real "understanding" inside, or just pattern matching so good it's indistinguishable from understanding — that's still an open question.

**What we DO understand vs what we DON'T.**

Think of a recipe versus a restaurant. We wrote the recipe — the code, the maths, the training procedure. We know every ingredient and every step. Nothing mysterious there.

But when you train on trillions of words, something happens inside those billions of numbers that we can't easily inspect. It's like following a recipe and getting a dish far better than expected, without knowing *which* mix of spices did it. You know the ingredients; you just can't pinpoint which interactions produced the result.

More concretely:

- **What we understand:** the maths. Tokens go in, get multiplied through layers of numbers, a prediction comes out. It's addition and multiplication at heart. No mystery.
- **What we don't understand:** *what* those billions of numbers have collectively learned to represent. We can't open the model and say "this group handles logic, this one French, this one sarcasm." It's all tangled together.

This is the **interpretability problem** — figuring out what a model's numbers actually mean. Researchers have found a few individual neurons that fire for specific concepts (one activates for the Golden Gate Bridge). But explaining the *whole* model is like understanding a city by looking at every brick. We know what bricks are; we just can't explain the city from that level.

**Bottom line.** We built the machine and we know how it runs. We just can't fully explain why it's this capable — because the capability comes from scale, billions of numbers interacting, not from any single clever design choice we can point to.

---

### How a transformer turns tokens into a prediction

Let's trace exactly what happens when the model reads `The cat sat on the` and predicts the next word. The *same* machinery runs during training and during chat — only what happens *after* the prediction differs (see step 8).

**First, the biggest misconception to clear up: the vocabulary exists *before* training.**

The vocabulary — the fixed list of (say) 50,000 tokens the model can read or produce — is built once, from the whole corpus, *before any training begins*. Every common word, `mat` included, is already on that list on day one. So the model *can* output `mat` from its very first step; it just gives it a tiny probability at first. Training never adds or removes words — it only changes the **probability** the model assigns to each word in a given context.

- Before training: `The cat sat on the ___` → `mat` ≈ 0.00002 (random, like every other word).
- After training: `The cat sat on the ___` → `mat` ≈ 0.65.

The word was always available; training just learned *when* to make it likely.

**1. Tokens and IDs**

A **token** is a chunk of text (often a sub-word piece). The tokenizer splits your text into tokens and gives each one an integer **ID** — a row number in that fixed vocabulary:

```text
"The cat sat on the"  →  [The][ cat][ sat][ on][ the]  →  IDs [464, 3797, 3332, 319, 262]
```

An ID carries no meaning. It is an *address* — "row 262 of the vocabulary" — like a seat number, not the person sitting in the seat.

**2. Token vectors**

There is one giant lookup table, the **embedding matrix**, with **one row per vocabulary word**. Each row is a list of numbers — that word's starting **vector**. Every row has the same length, the model's **hidden size** (e.g. 4,096 numbers). "Assigning a vector" to a token is just a lookup: ID 262 → grab row 262. Nothing clever.

```text
Embedding matrix (toy hidden size = 4)
  row 262 (the)  → [ 0.1, -0.3,  0.7,  0.2 ]
  row 319 (on)   → [-0.5,  0.4,  0.1,  0.0 ]
  row 3332 (sat) → [ 0.2,  0.2, -0.6,  0.9 ]
```

The facts that usually confuse people:

- **The embedding matrix *is* part of the parameters (the "weights").** Those rows are learned numbers, exactly like the attention and feed-forward weights. Toy: 50,000 words × 4 = 200,000 numbers. Real: 50,000 × 4,096 ≈ **205 million** — and that is *only* the embedding table. The attention, feed-forward, and output matrices add the rest, which is how you reach billions of parameters.
- **Random at start, better with training.** Every row begins as noise. Backprop nudges the rows until words used alike sit near each other (`cat` near `dog`, far from `Tuesday`). A half-trained model has half-useful embeddings; a finished one has good ones.
- **The vector size never varies.** Every token, in every input, gets a vector of the *same* fixed length (the hidden size). Only the numbers inside differ — never the length.
- **Same word → same *starting* row**, but attention (step 4) reshapes it by context, so the *final* vector for `the` in one sentence differs from `the` in another.

**3. Positional information**

Attention by itself is order-blind — without help, `dog bites man` and `man bites dog` look like the same bag of tokens. So the model adds **position information** to each vector. Older transformers add a position vector; many modern LLMs use **RoPE**, which rotates the attention query/key vectors by position. Either way, each vector now also knows *where* it sits and how far it is from the others.

**4. Transformer blocks: attention + feed-forward**

This is the heart of the model. A **transformer block** does two things, and the model stacks many of them (e.g. 32 or 96).

First, **attention** — each token gathers information from earlier tokens, via a matching game. Each token produces three things from its vector:

- a **query** = "what am I looking for?"
- a **key** = "what do I offer?"
- a **value** = "the actual information to hand over."

A token compares its query against every earlier token's key; where they match well, it copies in more of that token's value. So `sat` (a verb hunting for a subject) matches `cat`'s key (a noun/subject) and pulls in "my subject is cat." This is also **where contextual meaning is built**: `bank` next to `river` ends up with a different vector than `bank` next to `money`, even though both started from the same embedding row. A **causal mask** blocks a token from looking *right* (at future tokens) — right is the very thing we are predicting, so peeking would be cheating.

Second, **feed-forward** — the same small network runs over each token's vector on its own, refining whatever attention just gathered.

In short: attention *shares* information between tokens; feed-forward *digests* it. Stack the pair many times and the vectors grow steadily more context-aware.

**5. The number of vectors never changes — and why the *last* one makes the prediction**

Five input tokens produce five vectors. Every block still outputs five vectors of the same size; only the numbers inside change.

```text
The | cat | sat | on | the       <- 5 initial vectors
             transformer blocks
The | cat | sat | on | the       <- 5 contextual vectors
```

**These contextual vectors are temporary — they are never copied back into the embedding matrix.** During training, the prediction error is propagated backward through them, and the optimizer uses the resulting gradients to adjust the embedding rows (along with the model's other weights). During inference, there is no backward pass, so the embedding matrix does not change.

There is **no vector for the blank** in `The cat sat on the ___`. So which vector predicts the next word? The rule the model is trained on: **each position's vector is responsible for predicting the token that comes *after* it.** `The`'s position predicts `cat`; `cat`'s predicts `sat`; … and the **last token's** position (`the`) predicts whatever fills the blank. That is why we read off the top-of-stack vector of the *last real token* — by now it has absorbed the whole sentence, and predicting what's next is precisely its job.

**6. From the last vector to the next word**

1. Multiply that last vector by the **output matrix** → one raw score (a **logit**) for *every* word in the vocabulary (all 50,000).
2. Run **softmax** → squash those scores into probabilities that sum to 1.

```text
mat    0.65
floor  0.12
chair  0.05
...    (all 50,000, summing to 1.0)
```

- **The output is always from the vocabulary** — one entry from that 50,000-word list, never anything outside it. (Longer "new" words are built by stringing sub-word tokens together, but each piece is from the vocab.)
- **The whole distribution is computed** (a probability for every word), but **one** token is chosen.
- **The chosen word does *not* get a vector during this pass.** The output is just probabilities. `mat` only gets its embedding *later*, when it is appended and fed back in (step 7).
- **Greedy vs. variability:** you can always take the highest (`mat`) — *greedy decoding* — or roll a weighted die so the output varies. That choice is the whole subject of the next section.

**7. The generation loop, run by run (KV cache, and why the model has no memory)**

Generation is a loop — one token per step:

- **Run 1:** feed `The cat sat on the` → last vector → probabilities → pick `mat` → append → `The cat sat on the mat`.
- **Run 2:** the input is now `The cat sat on the mat`. *Here* `mat` finally gets its embedding and flows through the model. Read off the new last vector → predict the next token → append. Repeat until a "stop" token.

Two common questions:

- **KV cache.** In run 2, the keys and values for `The cat sat on the` were already computed in run 1. Recomputing them is wasteful, so they are saved in the **KV cache** and reused; only the new token's key/value is computed. Pure speed-up — the maths result is identical.
- **The model is stateless between separate requests.** The KV cache lives only *within one ongoing generation*. A brand-new request starts fresh with nothing remembered. A chatbot only "remembers" earlier messages because the app **resends the whole conversation as text** each time. The model's only memory is the text currently in front of it.

**8. Training run vs. inference run**

Same forward pass; the difference is what happens around it.

| | Training run | Inference run (you chatting) |
|---|---|---|
| Predicts at | *Every* position at once (parallel) | Effectively just the next token |
| Knows the answer? | **Yes** — the real next word is right there in the text | No — it is generating it |
| Afterwards | Measure error, **backpropagate, update all weights** (embeddings included) | **Nothing updated** — weights are frozen; just sample and loop |

So in training on `The cat sat on the mat`, the model makes all five predictions at once (`The`→`cat`, `The cat`→`sat`, … `The cat sat on the`→`mat`). For the last one, the true answer `mat` is sitting in the text *for free* — no human labels it. If the model gave `mat` only 0.02, that gap is the error; backprop nudges the weights so next time `mat` is more likely *in this context*. Repeat across billions of such pairs and the model becomes good at prediction. (A full numbered example with real loss numbers is in **A complete training example, end to end**, below.)

**Why attention replaced RNNs and LSTMs**

Earlier sequential models (**RNNs** and **LSTMs**) read tokens one at a time and carried a single running summary of the past. They did not only see the previous word, but distant information had to survive inside that constantly rewritten summary, and it could fade. Two transformer wins fixed this:

- **Direct access** — every token reaches *any* earlier token at full strength, not through a fading summary.
- **Parallel training** — all positions are processed together instead of one-by-one, which is the real reason large models became trainable. (Generation is still sequential, because each new token depends on the ones already produced.)

---

### How does the model pick the next token? Why isn't the output always the same?

The final layer of a transformer outputs a **probability distribution** — a probability score for every possible next token across the *entire vocabulary*. The question is how to turn those scores into one chosen token.

The obvious choice — always pick the highest-scoring token — is called **greedy decoding**, and it usually backfires. The text comes out repetitive and flat, and the model often gets stuck in a loop, repeating the same phrase. Adding a bit of controlled randomness gives more varied, natural-sounding text.

That randomness is tuned by **sampling strategies** — rules for choosing among the likely tokens.

**Temperature** — a knob that reshapes the probability distribution before a token is picked.

- `temperature = 0` → always pick the highest-probability token. Deterministic: same input gives the same output every time. Good when you want consistency, like extracting structured data.
- `temperature = 1` → sample straight from the model's raw distribution. A natural balance.
- `temperature = 2` → flatten the distribution, so low-probability tokens get a bigger chance. More creative and random, and more likely to go off the rails.

**Top-k sampling** — keep only the `k` most probable tokens, drop the rest, and sample from those.

**Top-p (nucleus) sampling** — keep the smallest set of tokens whose probabilities add up to at least `p` (say, 0.9), and sample from those. It is adaptive: that might be 3 tokens or 30, depending on how confident the model is.

**Why add randomness at all?**

1. **Creativity** — for writing, brainstorming, and dialogue, you want variation.
2. **Avoiding loops** — greedy decoding tends to repeat itself, because a strong prediction reinforces what the model just said.
3. **Exploration** — the top token can be good locally but lead to a dead end. Picking the second-best now and then lets the model find a better overall answer.

**Practical defaults:**

- Code generation, structured extraction → `temperature = 0` (you want predictability)
- Chat, creative writing → `temperature = 0.7–1.0` (natural variation)
- Brainstorming, idea generation → `temperature = 1.0–1.3` (more diverse outputs)

This is why ChatGPT gives slightly different answers each time to the same question — it's sampling, not greedy decoding.

---

### Then what are "reasoning models"? How is that different from normal models?

If every model just predicts the next token, what makes "reasoning" or "thinking" models (OpenAI's o1/o3/o4, DeepSeek-R1) different from standard chat models (GPT-4o, Claude Sonnet)?

**It's still next-token prediction. The only difference is *what* tokens the model is trained to produce.**

A standard chat model is trained to produce the **answer directly**. Ask "what's 17 × 24?" and it's optimised to output "408" straight away.

A reasoning model is trained to first produce a **chain of intermediate "thinking" tokens**, then the answer. Ask the same question and it writes something like:

> "Let me break this down. 17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408."

Those intermediate tokens aren't decoration. They serve as **working memory** — a scratch space the model writes for itself.

**Why intermediate tokens matter mechanically**

An LLM has no scratchpad, no RAM, no hidden workspace. During generation, the *only* memory it has is the sequence of tokens already produced. Each new token is predicted from everything before it.

So if a problem needs 5 logical steps and you force the model to jump straight to the answer in one token, it must do all 5 steps inside a single forward pass through the network. That's often too much — the network isn't deep enough to compute all of it internally.

Letting the model write the steps out as tokens fixes this. Each step becomes part of the context for the next one. The model **offloads its computation into the token stream**: step 3 can use the result of step 2 because step 2 is sitting right there in the context. This is why chain-of-thought works — more tokens means more compute time, which means more reasoning steps the model can chain together.

**How reasoning models are trained differently**

- **Reasoning traces in the data** — Instead of (question, answer) pairs, the training data uses (question, step-by-step reasoning, answer). The model learns to produce the steps as part of its output.
- **Reinforcement learning on reasoning quality** — The model is rewarded not just for the right final answer, but for reasoning chains that *lead* to right answers. A common method is outcome-based RL: generate many traces, keep the ones that reach the correct answer, and reinforce those.
- **Test-time compute scaling** — Reasoning models are built to spend more compute when answering (at inference time) by generating longer chains of thought. Standard models are tuned to be fast and concise; reasoning models deliberately trade speed for accuracy by "thinking longer."

**The "Thinking..." block in the UI**

When ChatGPT shows an expandable "Thinking..." block, those are the intermediate reasoning tokens the model generated before its final answer. Some providers hide these (you pay for them but don't see them); others show them. Nothing about the architecture is different — the model is still predicting the next token. It has simply been trained to predict *reasoning-shaped* tokens before *answer-shaped* ones.

**When to use which**

| Task type | Better model |
|---|---|
| Quick factual questions, creative writing, simple coding | Standard model (faster, cheaper) |
| Multi-step math, complex logic puzzles, hard coding problems | Reasoning model |
| Tasks where you need to show your work or verify correctness | Reasoning model |
| High-volume, low-latency applications | Standard model |

**The key takeaway:** reasoning models have no different architecture and no special reasoning engine bolted on. They're the same transformer doing the same next-token prediction. They've just been trained to think out loud — producing tokens that act as working memory — and this measurably improves accuracy on hard problems. It's still token prediction, only now it's prediction *about the reasoning process itself* rather than a jump straight to the conclusion.

---

### How does training actually happen? Is it unsupervised? What about RLHF?

Training runs in three stages, each using a different learning method:

1. **Pretraining** — the model reads huge amounts of text and learns to predict the next word.
2. **Supervised fine-tuning (SFT)** — it learns to respond like a helpful assistant.
3. **RLHF** — it learns to produce the responses humans prefer.

All three do the same thing under the hood: nudge the model's numbers to be slightly better. Only the meaning of "better" changes.

#### Stage 1 — Pretraining (self-supervised learning)

Start with what a **label** means: the correct answer you want the model to learn. To train a model to recognise cats, you show it a photo and the label is "cat" — and a human had to write that label by hand. That is **supervised learning**: expensive, slow, and it does not scale.

Self-supervised learning is the trick that makes LLMs possible: **the next word in a sentence is the label, and it is already sitting there in the text.** Nobody has to annotate anything.

Step by step:

1. Take a sentence from the internet: "The capital of France is Paris".
2. Hide the last word. Show the model: "The capital of France is ___".
3. The model guesses. Maybe it says "London" (wrong).
4. Reveal the real word: "Paris".
5. The model compares its guess to "Paris" — that is the label, free and already in the text.
6. It nudges its internal numbers so that next time it is likelier to guess "Paris" here.
7. Move to the next sentence and repeat.

Do this for trillions of sentences and the model gets very good at predicting what comes next. To win at that game it *has* to learn grammar, facts, logic, and how code works — because all of those decide which word comes next.

**Under the hood: vocabulary, batches, and masking**

**Vocabulary.** Before training starts, you build a **vocabulary**: a fixed list of all the tokens the model knows. A token is not always a whole word — it is a sub-word chunk produced by **Byte-Pair Encoding (BPE)**, an algorithm that scans a large corpus and merges the most frequent character sequences until the vocabulary hits a fixed size (typically 32K–200K tokens). Examples from GPT-4's tokeniser:

- `"hello"` → 1 token
- `"antidisestablishmentarianism"` → 6 tokens
- `" the"` (with a leading space) is a *different* token from `"the"` (no space)
- non-English text often costs more tokens per word

The model's output layer produces one probability per vocabulary entry, so vocabulary size sets the width of the output. The vocabulary never changes after training.

**Flatten first, then chop into rows.** The whole training corpus is concatenated into one long token stream, with `<|endoftext|>` inserted at every document boundary. Nobody types it by hand — the preprocessing pipeline appends it automatically at the end of each document, and it is a reserved special token defined when the tokeniser is built. For example:

```
[The] [cat] [sat] [on] [mat] [<|endoftext|>] [Dogs] [love] [to] [run] [<|endoftext|>] [Sky] [is] [blue] ...
```

This flat stream is chopped into fixed-length chunks — the **context length**, e.g. 2048 tokens. Each chunk becomes one row of the training tensor:

- **Rows = the batch dimension.** Each row is an *independent* training example. The model never mixes information across rows: row 2 has zero visibility into row 1, even if they sat next to each other in the flat stream. Batching is purely a GPU-efficiency trick.
- **Columns = the context/sequence dimension.** This is the position within one example.

Because `<|endoftext|>` lands wherever a document happened to end, it can appear anywhere in a row — start, middle, or end. It is not tied to any column.

**Causal masking — the hard rule.** When predicting the token at column `t`, the model may only see columns `0` through `t-1` in the same row. This is **causal masking** (or autoregressive masking), enforced by the architecture itself. You cannot look right, because right is the future — the thing you are trying to predict. Letting the model peek ahead would leak the answer.

**`<|endoftext|>` is a soft learned reset, not a hard wall.** Here is the subtle part. Causal masking lets the model see *every* token to its left in the row — including tokens from a previous document that happen to sit left of an `<|endoftext|>`. The architecture does not cut the leftward view at `<|endoftext|>`. In the row above, when predicting `"love"` the model technically sees `[The, cat, sat, on, mat, <|endoftext|>, Dogs]`, doc-1 tokens included. What it *learns through training* is to ignore everything before `<|endoftext|>`, because that earlier context never helps predict the next word in a new document. So:

- **Hard rule (architecture):** always look at everything to the left in this row.
- **Soft behaviour (learned):** treat `<|endoftext|>` as a wall; attention weights for tokens before it drop near zero.

**"Doesn't chopping split sentences?"** Yes. A phrase like `"Dogs love to run"` can straddle a chunk boundary and end up split across two unrelated rows. This is accepted noise. The model is not memorising specific sentences; it is learning statistical patterns across trillions of tokens. Any common phrase appears thousands of times and lands fully inside a chunk in nearly all of them. The few split copies do not matter at scale.

**Training chunking is not inference.** Fixed-row chunking is only a training-time step to make uniform GPU batches. When you chat with the model there is no chunking — your prompt is one continuous context, up to the model's context window. The model never sees rows or batches while serving you.

#### Won't it just memorise the answers?

Good question. If the model saw "The capital of France is Paris" only once and you tested it on that exact sentence, that would be memorising — like solving past exam papers. But that is not what happens.

The model sees "Paris" in *thousands* of contexts: "I flew to Paris last summer", "Paris is known for the Eiffel Tower", "The meeting was held in Paris, France". It does not learn "after 'The capital of France is', say Paris". It learns something deeper — that "Paris" connects to France, to capitals, to Europe, to cities, across many sentence shapes. It learns the *relationship*, not one sentence.

There are also far fewer numbers (parameters) than training examples. GPT-3 has 175 billion parameters but trained on 300 billion tokens. You cannot store 300 billion things in 175 billion numbers, so the model is mathematically forced to *compress* — to find patterns instead of storing facts. It is like studying for a 1000-question exam with only one page of notes: you cannot write all 1000 answers, so you have to learn the underlying principles.

#### What are parameters?

**Parameters = weights = the numbers inside the model.** Same thing, three names. "GPT-4 has 1.8 trillion parameters" means the model is 1.8 trillion individual numbers sitting in a file.

Before training, those numbers are random and meaningless. Training adjusts them, one tiny nudge at a time, until together they encode useful knowledge. After training, those number values *are* the model — there is nothing else.

When you download a model (say Llama 3 from Meta), you are literally downloading a file of numbers. A 70 billion parameter model at 2 bytes per number is about 140 GB. Run those numbers through the transformer architecture — the code that does the math on them — and you get a working LLM.

So "parameters" always just means the numbers the model learned during training. More parameters = bigger model = more room to store patterns = generally smarter, but also more expensive to run.

The model is not memorising sentences. It is learning patterns about language and the world, which let it handle sentences it has never seen — including ones that did not exist when it was trained.

#### Where is the knowledge stored?

In the **weights** — the billions of numbers that make up the model. That is it. No separate database, no hard drive of facts it looks things up in. Everything the model "knows" lives in the specific values of those numbers.

Think of a trained musician. They do not keep songs in a filing cabinet in their head. Their knowledge of music is spread across neural connections — how their fingers move, their sense of rhythm, their feel for harmony. Ask "where is the music stored?" and the answer is everywhere and nowhere in particular: it is in the *pattern* of connections, not one spot.

LLMs work the same way. "Paris is the capital of France" is not in one number. It is spread across thousands of numbers that together encode how "Paris", "France", "capital", and "Europe" relate. Nudging any one of those numbers might slightly weaken the model's grasp of French geography, but no single number *is* that fact.

This is why you cannot open an LLM and search it like a database. The knowledge is distributed, compressed, and tangled across billions of numbers. It is also why models sometimes get things slightly wrong — the compression is lossy, like a JPEG losing detail to save space.

#### Is it the order of the numbers or their magnitude? And do we know why a value is what it is?

Both matter. Each parameter has a **position** (it sits between specific layers and neurons — its role in the architecture) and a **value** (its magnitude — what it learned). Change the position and you change which computation it takes part in. Change the value and you change the result of that computation.

Do we know *why* one number ended up at 0.0037 rather than 0.0041? In theory, yes. It is the result of training: billions of examples each nudged it a little, and it landed at 0.0037 because that value, combined with all the others around it, minimises prediction errors across the data. It is deterministic math, not random.

In practice, no human can explain why that exact value matters. It is like asking why a single grain of sand sits in its exact spot on a beach — technically the result of waves, wind, and physics, but nobody can give a meaningful explanation for any one grain. The system is just too large.

The honest summary: **we understand the process that produced the values (training), and we can verify they work (the model predicts well), but we cannot give a human-readable reason for why any individual number is what it is.** It is not mysterious, just too complex to narrate — like explaining the RGB value of every pixel in a photo. The process is understood; the individual outcomes are too numerous to explain one by one.

#### What you get after pretraining — the base model

Imagine someone who learned purely by reading the entire internet — every book, article, Reddit thread, and codebase. They absorbed all of it. Now you talk to them. What happens?

They do not *answer* you. They *continue* what you said, as if writing the next paragraph. Type "What is 2+2?" and they might reply "What is 3+3? What is 4+4?" — because online, questions are often followed by more questions (quiz pages, FAQ lists). They are not trying to help; they are just continuing the pattern.

This is the **base model**. It knows a lot but does not know it should answer you. It is a text-continuation engine, not a chat assistant — like a person who read every book in a library but was never taught to hold a conversation. OpenAI does not ship this to users. It needs stages 2 and 3 before it becomes the ChatGPT you actually talk to.

You see this distinction directly in Azure AI Foundry and Hugging Face. Browse the models and you often find two flavours of the same family:

- `Llama-3-70B` — the **base model**, pretraining only. It completes text; it does not follow instructions.
- `Llama-3-70B-Instruct` (or `-Chat`) — the same model after SFT + RLHF (stages 2 and 3). It answers questions and behaves like ChatGPT.

Not everyone ships the base model. OpenAI does not give you GPT-4 base, only the post-trained version. Meta (Llama), Mistral, and others do release base models publicly, because researchers want them for fine-tuning on their own data. For 99% of applications you want the `-Instruct` / `-Chat` variant; the base model only helps if you are doing your own SFT/RLHF.

#### Stage 2 — Supervised fine-tuning (SFT)

Human contractors — often thousands of them — write ideal conversations: "Given this user message, here is the ideal helpful response." The model trains on these (input, output) pairs using ordinary supervised learning. This teaches it the *format* of being helpful: answer directly, follow instructions, stay concise, use the right tone.

This is what turns a base model into something that behaves like a chat assistant.

#### Stage 3 — RLHF (Reinforcement Learning from Human Feedback)

The fine-tuned model generates several responses to the same prompt. Human raters compare pairs and mark which one is better — more helpful, more accurate, less harmful. They do not write new responses; they just rank existing ones.

These thousands of pairwise preferences train a separate **reward model** — a smaller model that learns to predict what a human would prefer. It outputs a score, e.g. "a human would rate this 7/10".

The LLM is then optimised with reinforcement learning — specifically **PPO** (Proximal Policy Optimisation) — to maximise the reward model's score. The LLM generates, the reward model scores, the LLM adjusts to score higher, and this loop repeats many times.

RLHF is how you steer the model toward being helpful, honest, and safe — qualities that are hard to write as explicit rules but easy for humans to recognise when comparing two options.

#### All three stages just adjust the same numbers

That is the punchline. There is no other mechanism. Every stage does the same thing — nudge the model's parameters to be slightly better. Only the definition of "better" changes:

| Stage | What "better" means | What changes |
|---|---|---|
| Pretraining | "Better at predicting the next word" | The same numbers |
| SFT | "Better at responding like a helpful assistant" | The same numbers |
| RLHF | "Better at producing responses humans prefer" | The same numbers |

It is one file of numbers refined three times, with three definitions of "good". Think of sculpting: pretraining carves the rough shape (knowledge), SFT refines the form (helpfulness), RLHF polishes the surface (safety, tone, preference). Each stage works the same block of stone — just with finer tools and a different goal.

After all three stages you have one file of numbers. That is ChatGPT (or Claude, or Gemini). Nothing else runs behind the scenes — no secret database, no rule engine, no internet connection. Just numbers being multiplied together very fast.

#### Can an instruct model be fine-tuned again after RLHF?

Yes. RLHF does not lock or finish the model; it just produces another checkpoint whose weights can be updated again. Enterprises usually start from an instruct model and apply more SFT — often through **LoRA/QLoRA** adapters — using examples of their own tasks, terminology, formats, or tool calls. Preference methods such as **DPO** or RL can also be applied again when preference data is available.

Further training can weaken general instruction-following or safety behaviour, so it normally uses a low learning rate, high-quality data, and **regression evaluations** (tests that check old skills did not break). Not every instruct model uses traditional RLHF; some use SFT followed by preference optimisation such as DPO.

---

### If it's all just numbers, is more parameters the only way to make models better?

No. That was the belief from roughly 2020–2023: just make it bigger and it gets smarter. And it worked. GPT-2 (1.5B parameters) → GPT-3 (175B) → GPT-4 (rumoured ~1.8 trillion) each jumped clearly in capability. This was the **scaling laws** era — the rule that more parameters + more data + more compute reliably means a better model.

But size is only one lever. There are several.

**The levers you can pull to improve a model:**

1. **More parameters** — Bigger model, more capacity. But returns shrink and cost grows fast (training GPT-4 reportedly cost $100M+).

2. **More or better training data** — A smaller model trained on clean data can beat a bigger model trained on garbage. This is where things recently shifted: companies now carefully curate and filter data instead of scraping everything.

3. **Better architecture** — The Transformer (2017) was a huge leap. Smaller tweaks still pay off: better attention mechanisms, or **mixture-of-experts**, where only part of the model activates per token. You get gains without adding raw size.

4. **Better training recipes** — Learning rate schedules, data ordering, curriculum (easy examples first, hard later), longer training on the same data. Much recent progress comes from training the *same size* model more carefully.

5. **Post-training improvements** — Better RLHF, better fine-tuning data, and **DPO** (Direct Preference Optimisation, a simpler alternative to RLHF). Cheap next to pretraining, but it gives big quality bumps.

6. **Test-time compute** (reasoning models) — Instead of a bigger model, let it *think longer* when answering. Reasoning models like o1 and o3 do this: roughly the same model, but it generates more tokens (a chain of thought) before answering, trading speed for accuracy.

**Where are we hitting walls?**

- **Data.** The biggest wall. Models have been trained on nearly all publicly available human writing, so you can't just "get more." Companies now buy book licences, generate **synthetic data** (use models to create training data for other models), and tap video and audio.

- **Cost and energy.** A frontier model costs $100M–$1B in compute and uses the electricity of a small city for months. Doubling parameters roughly quadruples training cost, and there's a physical limit to how many GPUs you can wire together and power.

- **Diminishing returns from scale.** Each 10x in size helps less than the last. 7B → 70B is a huge jump; 70B → 700B is smaller; 700B → 7 trillion is smaller still. Scaling forever won't keep progress linear.

- **Inference cost.** Even if you could train a 10-trillion-parameter model, running it would be too slow and expensive for real products. Users expect answers in seconds, not minutes.

**Where progress is actually happening now (2025–2026):**

| Direction | What it means |
|---|---|
| **Smaller, smarter models** | Train smaller models better (better data, longer training, distillation from bigger models). Llama 3 8B rivals GPT-3.5, which is 20x larger. |
| **Reasoning at inference time** | Let models think longer on hard problems. Costs more per query, but needs no retraining. |
| **Synthetic data** | Use existing good models to generate training data for new ones. Controversial but working. |
| **Mixture of Experts (MoE)** | Many "expert" sub-networks, but only a few activate per token. The capacity of a huge model at the inference cost of a small one. |
| **Multimodal training** | Train on images, video, and audio, not just text. New data sources that aren't exhausted yet. |
| **Better post-training** | RLHF improvements, DPO, constitutional AI. Get more from the same base model through better alignment. |
| **Longer context windows** | Process more text at once (100K+ tokens). Doesn't make models smarter, but far more useful for real tasks. |

So: more parameters was the easy answer from 2020–2023. It still helps, but we've hit diminishing returns and cost walls. Progress now comes from being *smarter* — better data, better recipes, better architectures, reasoning at inference — rather than brute-forcing scale. The field shifted from "just make it bigger" to "make the most of what we have."

---

### What is backpropagation? Are human brains like this?

**What a neural network is.** A neural network is just layers of maths, joined by **weights** — plain numbers, often billions of them. Input data flows forward through the layers, and each layer transforms it a little, until the final layer produces an output. That forward flow is the **forward pass**.

**What backpropagation does.** Backpropagation ("backprop") is how the network learns from its mistakes. It runs four steps:

1. **Forward pass** — push an input through the network and get an output (say, a predicted next token).
2. **Calculate loss** — compare that output to the right answer. The gap is the **loss**: one number saying how wrong the model was.
3. **Backward pass** — using calculus (the chain rule for derivatives), trace backward through every layer to see how much each weight added to the error. Each weight gets a **gradient**: a direction and size that says "nudge this weight up and the error goes up/down by this much."
4. **Update weights** — shift each weight a little in the direction that lowers the error. This step is **gradient descent**.

Repeat the cycle billions of times over the training data. The weights slowly settle into values that make the fewest prediction errors.

**The dial analogy.** Picture a machine with a million dials. It spits out the wrong answer. Backprop tells you which dials caused most of the error, and which way to turn each one to fix it. Do that millions of times and the machine becomes accurate.

**So how does it know which dial is the culprit?** The honest answer is **calculus** — the chain rule from school, applied automatically to every weight.

The model is one giant nested function: input → layer 1 → layer 2 → ... → layer 96 → output. Each layer multiplies by some weights and applies a simple function. The whole network is a single huge expression with billions of variables (the weights).

When we get a loss number (say the model predicted "London" but the answer was "Paris"), we want to know, for each weight: **if I nudge it up a tiny bit, does the loss go up or down, and by how much?** That "how much the loss changes when I change this one weight" is the **gradient** of the loss for that weight. It is a derivative.

The chain rule says that if `y = f(g(h(x)))`, the derivatives multiply backwards through the chain. A neural network is exactly that — a long chain of functions. So the gradient for a weight in layer 1 = (derivative of loss w.r.t. layer 96 output) × (derivative of layer 96 w.r.t. layer 95) × ... × (derivative of layer 2 w.r.t. layer 1). One long product, multiplied backwards from output to input.

**The backward pass, step by step:**

1. **Forward pass.** Run the input through the network. Save every layer's output along the way.
2. **Compute loss.** Compare the final output to the right answer. Get one number.
3. **Gradient at the output.** Easy — the derivative of the loss w.r.t. the output is a simple formula.
4. **Propagate backward, layer by layer.** Each layer already has the gradient flowing in from the layer ahead. Multiply by that layer's own derivative and pass the result back to the previous layer. The chain rule handles the bookkeeping.
5. **A gradient per weight.** Each one says: "increase this weight by ε and the loss changes by about (gradient × ε)."

So "how does it know who's to blame?" isn't a human judgement — the math just falls out. A weight that swung the wrong output gets a big gradient; a weight that barely mattered gets a tiny one. Not "this weight is guilty," but "changing this weight by 1 unit moves the loss by 0.0003, while that one moves it by 0.07 — so the second one is more responsible, adjust it more."

**Updating the weights.** Each weight then changes by `new_weight = old_weight - learning_rate × gradient`. The minus sign means "move opposite to the gradient": the gradient points toward *more* loss, and we want less. The **learning rate** is a small number (e.g. 0.0001), so each step is tiny and we don't overshoot.

**Frameworks do the calculus for you.** PyTorch and TensorFlow handle all of this through **autograd** (automatic differentiation). You write only the forward pass; the framework records every operation and works out the backward pass itself. You never compute a derivative by hand. That is why deep learning became practical — you don't need to be a calculus wizard to train a model.

**Are human brains like this?** They look similar but work very differently:

| Aspect | Neural Networks | Human Brains |
|---|---|---|
| Basic unit | Artificial neuron (mathematical function) | Biological neuron (electrochemical cell) |
| Connections | Weights (numbers adjusted by backprop) | Synapses (strengthened by repeated firing — Hebbian learning: "neurons that fire together wire together") |
| Learning signal | Explicit error signal propagated backward | No known backward error signal — brains likely use different local learning rules |
| Data efficiency | Needs billions of examples | Learns from very few examples (a child sees a dog 3 times and generalises) |
| Timing | Trained once, then frozen | Learns continuously throughout life |
| Embodiment | None — processes text/numbers | Fully embodied — tied to senses, emotions, survival instincts, motor control |

The brain inspired the *idea* of neural networks — layers of connected units that learn by adjusting connection strengths. But backprop itself has no known biological match. Brains don't run calculus on error gradients; they use local learning rules (each synapse adjusts from nearby activity) and are shaped by evolution, not by an optimisation algorithm. Use the analogy for intuition, not as literal truth.

---

### A complete training example, end to end

Let's walk through one training step using everything above. The example is tiny so the numbers stay followable, but the principles scale exactly to GPT-4.

**Setup:**

- Vocabulary: 50,000 tokens
- Model: small transformer, say 100M parameters, 12 blocks
- Training text snippet: `"The capital of France is Paris"`
- After tokenisation: `[The] [ capital] [ of] [ France] [ is] [ Paris]` — 6 tokens with IDs say `[464, 3139, 286, 4881, 318, 6342]`

**Step 1 — Prepare input and target.** The model learns to predict the next token at every position. From this one sentence we get 5 prediction tasks at once:

| Position | Input (sees) | Target (must predict) |
|---|---|---|
| 1 | `The` | ` capital` |
| 2 | `The capital` | ` of` |
| 3 | `The capital of` | ` France` |
| 4 | `The capital of France` | ` is` |
| 5 | `The capital of France is` | ` Paris` |

All 5 predictions happen in **one forward pass**, thanks to causal masking — each position only sees the tokens to its left.

**Step 2 — Forward pass.**

1. Each token ID becomes an **embedding** — a learned vector of size, say, 768. We now have a 6×768 matrix.
2. **Positional information** is applied (for example, with RoPE) so the model knows the token order.
3. The embeddings pass through **transformer block 1**: attention (tokens look left and gather info), then feed-forward (each token's vector is refined).
4. Block 1's output → block 2 → block 3 → ... → block 12.
5. After the final block, we still have a 6×768 matrix.
6. The **output projection** layer multiplies it by a 768×50,000 matrix, giving a 6×50,000 matrix of **logits** (raw scores, one per vocabulary token).
7. **Softmax** turns each row of logits into probabilities that sum to 1 — a probability distribution over the whole vocabulary.

So at position 5, after seeing `"The capital of France is"`, the model gives probabilities for all 50,000 possible next tokens. Maybe:

- `Paris` → 0.42
- `the` → 0.08
- `Lyon` → 0.05
- `London` → 0.03
- ...everything else summing to 0.42

**Step 3 — Compute loss.** For each of the 5 positions, compare the predicted distribution to the real answer using **cross-entropy loss**. It is high when the model gave the correct token low probability, and low when it gave it high probability.

The formula's shape: `loss = -log(probability assigned to correct token)`.

At position 5, the model gave `Paris` probability 0.42 → loss = `-log(0.42) ≈ 0.87`.

Had it given `Paris` 0.99, the loss would be `-log(0.99) ≈ 0.01` — almost nothing. Had it given 0.001, the loss would be `-log(0.001) ≈ 6.9` — large.

Average the loss across all 5 positions → one number, e.g. `1.34`. That's the total loss for this example.

**Step 4 — Backward pass.** The framework (PyTorch, etc.) traces backwards through every operation that fed into the loss. For each of the 100 million weights it computes a gradient via the chain rule (as in the backprop section above):

- A weight in block 7's attention that shaped how `France` influenced position 5: gradient = 0.003
- A weight in the embedding for the token `Paris`: gradient = -0.012
- A weight in some unrelated part of the network: gradient = 0.0000001

Autograd does this automatically. No human writes derivative formulas.

**Step 5 — Update weights.** For each weight: `new_weight = old_weight - learning_rate × gradient`. With learning rate = 0.0001:

- Weight with gradient 0.003 → new value = old - 0.0000003 (tiny nudge down)
- Weight with gradient -0.012 → new value = old + 0.0000012 (tiny nudge up)

Each weight moves a hair's breadth. On its own, meaningless.

**Step 6 — Repeat.** Next training example. Do it all again. And again.

**Scale.** Each batch processes thousands of examples in parallel on GPUs, and a full run processes trillions of tokens. For a frontier model this takes months on tens of thousands of GPUs. After all those billions of tiny weight nudges, the weights settle into values that minimise the loss across the whole training corpus. That accumulated competence is what we experience as "the model knows things."

**Key insight.** Every chat you have with ChatGPT is just the *inference-time* version of step 2 — the forward pass. The model isn't learning from your conversation; it's running its already-trained weights forward to produce probabilities, sampling from them (see the temperature/top-p section), and outputting tokens. All the learning happened during training. Inference is frozen.

---

### Why do LLMs hallucinate?

Because their job is to produce **plausible-sounding text**, not **verified true text**. Those are not the same thing, and nothing in training guarantees truth.

Five root causes:

**1. The prediction mechanism.** The model picks whatever token is statistically most probable given the context. "Most probable" is not the same as "correct." A citation written in academic style can be entirely made up — because that is what citation-shaped text looks like, even if the paper does not exist.

**2. No external lookup, and lossy memory.** When you ask a question, the model does not search a database or check Wikipedia. It generates purely from the compressed knowledge in its weights. That compression is **lossy** (some detail is dropped to save space). Where knowledge is missing, the gap gets filled with a plausible guess instead of "I don't know."

**3. No uncertainty signal.** The model writes every token the same mechanical way, whether it is reciting a well-known fact or inventing one. It has no reliable internal "I'm not sure" flag tied to how accurate that specific claim really is.

**4. Training-data quality.** The internet is full of misinformation, contradictions, outdated facts, and fiction told as fact. The model learned from all of it, with no way to sort reliable sources from unreliable ones during pretraining.

**5. Context pressure.** If the conversation clearly expects a specific name, date, or fact, the model is statistically pulled toward filling that slot with *something* — even if it has to fabricate it. Saying "I don't know" is less probable when the user obviously wants a concrete answer.

**How hallucination is reduced in practice:**
- **RAG** — Give the model real source documents at query time, so it answers from evidence rather than memory.
- **Lower temperature** — Temperature controls randomness in token selection. Lower temperature means the model sticks to its highest-probability tokens: less creative, but less likely to confabulate.
- **Chain-of-thought prompting** — Make the model reason step by step before answering, which exposes logical errors earlier.
- **Output validation** — A second model or a set of rules checks the first model's claims against known sources.
- **Grounding with structured data** — Wire the model to databases and APIs for factual queries, instead of trusting parametric memory (the knowledge baked into its weights).

---

### Why Is This Called Emergence?

Pattern learning itself was expected. To predict text well, a model benefits from learning grammar, concepts, cause and effect, proof structures, and common reasoning patterns. It does not store every written deduction separately. It learns compressed, distributed regularities that can be recombined for new problems.

In LLM research, a capability is called **emergent** when it has three traits:

- It was not explicitly programmed or directly targeted.
- It works far better in larger models than in smaller ones.
- It was hard to predict by extrapolating from smaller-model results.

Emergent does not mean magical or uncaused. Researchers expected next-token prediction to improve as models grew. What was less certain was that this single objective would also produce broad abilities: learning a task from examples in the prompt, following new instructions, writing programs, and combining skills across domains without separate training for each.

The term is debated, partly because of how we measure it. Some "sudden" abilities are an artefact of **binary scoring** (an answer counts as `0` unless it is fully correct, then `1`). An almost-correct answer still scores `0`. When you instead measure partial progress continuously, many capabilities improve smoothly rather than snapping on at one threshold.

Here is why binary scoring distorts the picture. If a solution needs five correct steps, steady per-step improvement can look like a sudden jump in finished solutions:

```text
60% reliability per step → 0.6⁵ ≈ 8% complete solutions
90% reliability per step → 0.9⁵ ≈ 59% complete solutions
```

So the careful reading is this: scaling gradually improves the model's internal representations and their reliability, but whole-task benchmarks can make the resulting capability look abrupt.

### Why Does Scaling Help?

Scaling means balancing **model parameters** (the model's numbers), **training data**, and **training compute** together, not just adding parameters. Five reasons larger scale helps:

1. **More representational capacity.** More parameters let the model represent more features and finer distinctions. A small model has to reuse limited capacity across many patterns, so they interfere with each other.
2. **Better composition.** Hard tasks require chaining several learned operations: understand a question, pick a rule, track variables, produce an answer. More layers, dimensions, and attention capacity make these combinations more reliable.
3. **Better coverage of rare patterns.** More data means more examples of uncommon facts, code, proofs, and specialised language. More parameters help the model keep these without crowding out common ones.
4. **Better predictive representations.** Continuing text correctly often means modelling entities, context, time, and cause and effect. Lower prediction loss therefore pushes the model toward more useful internal representations of language and the world.
5. **Higher end-to-end reliability.** Small per-step gains multiply across a multi-step task, producing a much larger gain in complete solutions.

Parameters alone are not enough. A model also needs sufficient high-quality data and compute to train them, plus good architecture, training objectives, post-training, and inference-time computation. Scaling reliably improves performance in practice, but researchers still cannot fully explain, mechanistically, every ability it produces.

---

## 8. Creating Small Language Models (SLMs) from LLMs

An **SLM (Small Language Model)** uses fewer parameters and less compute than a general-purpose LLM. It is often specialised for a specific task and offers lower cost, latency, and memory usage.

Two common techniques are:

- **Knowledge distillation:** trains a smaller student model using a larger teacher LLM.
- **Quantization:** stores an existing model's weights using fewer bits.

### Knowledge Distillation

A teacher LLM generates answers, labels, tool calls, or token probabilities. A smaller pretrained student model is fine-tuned to reproduce them.

1. Collect representative prompts for the target task.
2. Generate answers using the teacher LLM.
3. Filter incorrect, unsafe, and low-quality examples.
4. Fine-tune the student model.
5. Evaluate it on a separate held-out dataset.

| Method | Training signal |
|---|---|
| **Response distillation** | Teacher-generated text or structured outputs. Works with API-only models. |
| **Logit distillation** | Teacher token probabilities. Richer signal, but requires access to logits. |

Teacher outputs are synthetic data and may contain hallucinations or biases. Verify them before training, especially for reasoning and tool-use examples.

### Quantization

Quantization converts weights from formats such as FP32 or FP16 to INT8 or INT4. It reduces memory usage and may improve inference speed when supported by the hardware and runtime.

| Weight format | Approximate weight memory |
|---|---:|
| FP32 (32-bit) | 28 GB |
| FP16/BF16 (16-bit) | 14 GB |
| INT8 (8-bit) | 7 GB |
| INT4 (4-bit) | 3.5 GB |

These estimates cover weights only. Runtime memory also includes the KV cache and temporary activations. Quantization reduces storage precision, not parameter count.

| Approach | Description |
|---|---|
| **PTQ** | Quantizes an already trained model. Fast, but may reduce quality. |
| **QAT** | Simulates quantization during training. Better quality retention, but requires more training. |

### Combining Distillation and Quantization

```text
Teacher LLM → distil into student SLM → quantize SLM → deploy
```

Evaluate task accuracy, safety, latency, throughput, and memory after both distillation and quantization. Use a held-out dataset that was not used for training or quantization calibration.

---

## 9. Glossary

| Term | Definition |
|---|---|
| **LLM** | Large Language Model. A neural network trained on huge amounts of text that can generate, summarise, translate, and reason over language |
| **SLM** | Small Language Model. A lower-compute language model, often specialised for a narrower task or deployment environment |
| **Knowledge distillation** | Training a smaller student model to reproduce useful behaviour or token probabilities from a larger teacher model |
| **Quantization** | Representing model weights or computations with fewer bits to reduce memory use and potentially improve inference speed |
| **PTQ** | Post-Training Quantization. Converting an already trained model to lower precision, usually with little or no additional training |
| **QAT** | Quantization-Aware Training. Training while simulating low-precision arithmetic so the final quantized model retains more quality |
| **LCEL** | LangChain Expression Language. The pipe (`\|`) syntax for composing chains declaratively |
| **DAG** | Directed Acyclic Graph. A graph whose edges have direction and never loop back. LangChain's execution model |
| **Node** | A single computation step in a LangGraph graph, written as a Python function |
| **Edge** | A connection between nodes in LangGraph. Fixed (always followed) or conditional (chosen at runtime) |
| **State** | The shared typed object that every node in a LangGraph graph reads from and writes to |
| **Checkpointer** | Saves graph state after every node, so a run can resume or replay |
| **Thread ID** | A unique ID linking one graph run to its saved checkpoints |
| **ReAct** | Reasoning + Acting. An agent loop: reason → act → observe → repeat until done |
| **RAG** | Retrieval-Augmented Generation. Fetch relevant documents, drop them into the prompt, generate a grounded answer |
| **Embedding** | A vector (list of numbers) that captures the meaning of text. Similar meanings → similar numbers |
| **Vector Store** | A database built to store embeddings and find similar ones fast (e.g., Pinecone, Chroma, FAISS) |
| **Chunking** | Splitting big documents into smaller pieces so each piece gets a focused, precise embedding |
| **Token** | The unit an LLM reads text in. Roughly ¾ of a word. You pay per token for API calls |
| **Tool** | A function an agent can call at runtime (web search, calculator, API call, database query, etc.) |
| **Human-in-the-Loop** | A pattern where the run pauses for human review or approval before continuing |
| **Serialisation** | Turning an in-memory object into a storable format (JSON, bytes) and back |
| **Non-deterministic** | Giving different outputs for the same input on different runs. LLMs are non-deterministic by nature |
| **Idempotent** | An operation that gives the same result however many times you run it. Handy for safe retries |
| **Latency** | The delay between sending a request and getting a response |
| **Self-supervised** | Training where the labels come from the data itself (e.g., predict the next word). No human labelling needed |
| **Base model** | The raw model straight after pretraining. Knows a lot but only completes text — not yet an assistant |
| **SFT** | Supervised Fine-Tuning. Training on human-written ideal answers so the model acts like an assistant |
| **RLHF** | Reinforcement Learning from Human Feedback. Tuning outputs based on humans ranking which answers they prefer |
| **Reward model** | A model trained on human preferences that scores how good a response is. Guides RLHF |
| **Backpropagation** | The learning algorithm: measure the error, trace it backward to find which weights caused it, then adjust them |
| **Gradient descent** | The update step: nudge each weight in the direction that lowers error, sized by its contribution |
| **Loss** | A number measuring how wrong the prediction was. Training drives it down |
| **Weights/Parameters** | The billions of numbers inside the network that define its behaviour. Adjusted during training |
| **Emergence** | A capability nobody trained for that becomes much stronger at scale and was hard to predict from smaller models; the apparent *suddenness* often depends on how you measure performance |
| **Hallucination** | When a model produces plausible-sounding but false or made-up content |
| **Temperature** | Controls randomness in generation. Lower = more predictable. Higher = more creative/risky |
| **Diffusion model** | Architecture for image/video generation. Learns to strip noise from random static, guided by a text prompt |
| **Modality** | A type of data — text, image, audio, video. Multimodal models handle several at once |
| **Base model** vs **Instruct/Chat model** | Base = pretraining only, just completes text. Instruct/Chat = base + SFT + RLHF, acts like an assistant. Foundry lists both |
| **Vocabulary** | The fixed list of all tokens a model can read or produce. Set during tokenisation, never changes |
| **Tokeniser / BPE** | Byte-Pair Encoding. Builds the vocabulary by merging frequent character sequences into tokens |
| **Causal masking** | Rule that a token can only attend to earlier tokens. Stops the model cheating by seeing future tokens in training |
| **Attention** | Transformer layer where tokens look at each other and gather relevant info. Uses query/key/value vectors |
| **Feed-forward layer** | Transformer layer applied to each token on its own. Refines what attention gathered. Holds most factual knowledge |
| **Transformer block** | One unit of attention + feed-forward. Stacked many times (e.g., 96 in GPT-3) to build the full model |
| **RNN / LSTM** | Pre-2017 architecture. Processed text one token at a time. Replaced by transformers for parallelism and long-range handling |
| **Forward pass** | One run of input through the model to produce output. Inference = one forward pass per generated token |
| **KV cache** | Caches attention keys/values from earlier tokens so each new token skips recomputing the full context |
| **Softmax** | Turns a vector of raw scores into a probability distribution (all positive, summing to 1) |
| **Logits** | The raw pre-softmax scores from the output layer — one per vocabulary token. Softmax turns them into probabilities |
| **Positional encoding** | Order/distance info added to token vectors so attention knows where tokens sit. RoPE applies it by rotating query/key vectors |
| **Multi-head attention** | Several attention operations running in parallel in one layer, each learning a different kind of relationship (syntax, coreference, etc.) |
| **Cross-entropy loss** | The loss used to train language models. High when the model gave the correct token a low probability |
| **Gradient** | How much the loss would shift if you nudged one weight. Computed by backprop |
| **Learning rate** | The step size when updating weights. Kept small (e.g., 0.0001) to avoid overshooting the minimum |
| **Autograd** | A feature in frameworks like PyTorch that auto-computes gradients for any forward pass you define |
| **Sampling** | Picking a token from the model's probability distribution. Controlled by temperature, top-k, top-p |
| **Top-k / Top-p sampling** | Restrict sampling to the most likely tokens (top-k = fixed count, top-p = until cumulative probability ≥ p) |
| **Greedy decoding** | Always pick the highest-probability token. Deterministic but often repetitive |
| **LLM-as-a-judge** | Using one LLM to score or grade another model's outputs instead of a human reviewer |
| **Faithfulness / Groundedness** | Whether an answer is actually supported by the retrieved documents, not invented |
| **Golden dataset (eval set)** | A curated set of inputs with known good answers, used as the benchmark to test a pipeline against |
| **BLEU / ROUGE** | Automatic text-overlap scores comparing generated text to a reference (BLEU favours precision, ROUGE favours recall) |
| **METEOR** | A reference-based translation metric that gives credit for stems and synonyms and penalises fragmented word order |
| **Perplexity** | How surprised a language model is by held-out text; lower means better next-token prediction, not necessarily more correct answers |
| **Precision / Recall / F1** | Precision = how much of what you returned was correct; recall = how much of the correct stuff you returned; F1 = their balance |
| **RAGAS** | A library that scores RAG pipelines on metrics like faithfulness, answer relevance, and context quality |
| **Regression testing (for prompts)** | Re-running a fixed eval set after a prompt change to check nothing that worked before now breaks |
| **A/B testing** | Sending some traffic to version A and some to version B, then comparing quality on real users |
| **Offline vs online evaluation** | Offline = scoring against a fixed dataset before release; online = measuring quality on live traffic in production |
| **`<\|endoftext\|>`** | Special token marking document boundaries during training. Lets multiple unrelated documents share one row |
