
# MCP

LLMs in themselves are nothing more than next token predictors.

Pre-MCP, developers gave LLMs access to tools - think of them as api's. This was known as **function calling** (OpenAI) or **tool use** (Anthropic).

Developers defined the available tools JSON schema and sent this to the LLM with user message. LLM returns back a structured response which is executed by the application code and sent to the LLM for a final response. 


