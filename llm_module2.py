from typing import Optional, Sequence

import config
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START


class LLMService:
    """Minimal LangGraph-based chat service with per-thread memory and optional RAG context."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        system_template: Optional[str] = None,
    ) -> None:
        self._llm = ChatOllama(
            base_url=base_url or config.OLLAMA_HOST,
            model=model or config.OLLAMA_MODEL,
        )

        self._system_template = system_template or (
            "You are a helpful assistant.\n"
            "Use the conversation history to resolve references.\n"
            "Use any provided 'Context' message(s) for factual content.\n"
            "If the answer is not present in the context, reply exactly:\n"
            '"The answer is not available in the provided context."'
        )

        self._checkpointer = InMemorySaver()
        self._default_thread_id = "default"
        self._graph = self._build_graph()

    def chat(
        self,
        question: str,
        context: Optional[str | Sequence[str]] = None,
        thread_id: Optional[str] = None,
    ) -> str:
        """Ask a question with optional retrieved context. Memory is keyed by thread_id."""
        tid = thread_id or self._default_thread_id

        # Prepare input messages (context is optional and per-turn)
        messages_input: list[dict] = []
        if context:
            if isinstance(context, str):
                ctx_text = context
            else:
                ctx_text = "\n\n".join(context)
            messages_input.append({"role": "system", "content": f"Context:\n{ctx_text}"})
        messages_input.append({"role": "user", "content": question})

        result = self._graph.invoke(
            {"messages": messages_input},
            {"configurable": {"thread_id": tid}},
        )
        messages = result.get("messages", [])
        return messages[-1].content if messages else ""

    def llm_query(self, question: str, context: Optional[str | Sequence[str]] = None, thread_id: Optional[str] = None) -> str:
        return self.chat(question=question, context=context, thread_id=thread_id)

    def reset_memory(self) -> None:
        """Clear all conversation memory for all threads."""
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph()

    def _reset_memory(self) -> None:
        self.reset_memory()

    def set_system_prompt(self, new_template: str) -> None:
        """Replace the system prompt while preserving memory."""
        self._system_template = new_template
        self._graph = self._build_graph()

    def _set_system_prompt(self, new_template: str) -> None:
        self.set_system_prompt(new_template)

    def _build_graph(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self._system_template),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        def call_model(state: MessagesState):
            prompt_value = prompt.invoke({"messages": state["messages"]})
            response = self._llm.invoke(prompt_value)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("llm", call_model)
        builder.add_edge(START, "llm")
        return builder.compile(checkpointer=self._checkpointer)


'''if __name__ == "__main__":
    service = LLMService()
    print("LangGraph + Ollama chat (with memory). Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        try:
            # For demo purposes, context is empty; in app, pass retrieved chunks here
            ans = service.chat(q, context="")
        except Exception as exc:  # noqa: BLE001
            ans = f"Error: {exc}"
        print(f"Assistant: {ans}\n")'''


