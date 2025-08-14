from typing import Optional

import config
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START


class LLMService:
    """LangChain-based LLM service backed by local Ollama with conversation memory."""

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
            "You are a question-answering assistant.\n"
            "Use the conversation history to resolve references (e.g., subject/entity of the question).\n"
            "Use the provided context messages for factual content.\n"
            "If the answer is not present in the context, reply exactly:\n"
            '"The answer is not available in the provided context."'
        )
        self._checkpointer = InMemorySaver()
        self._thread_id = "default"
        self._graph = self._build_graph()
        self._printed_counts: dict[str, int] = {}

    def Llm_query(self, question: str, context: str, thread_id: Optional[str] = None) -> str:  # Public API
        """Generate an answer using short-term memory and provided RAG context.

        thread_id: optional identifier to isolate memory across concurrent users.
        """
        tid = thread_id or self._thread_id
        final_msg: AIMessage | BaseMessage | str | None = None
        last_count = self._printed_counts.get(tid, 0)

        # Inject context as a system message for this turn (if provided)
        msgs_input = []
        if context:
            msgs_input.append({"role": "system", "content": f"Context:\n{context}"})
        msgs_input.append({"role": "user", "content": question})
        for chunk in self._graph.stream(
            {"messages": msgs_input},
            {"configurable": {"thread_id": tid}},
            stream_mode="values",
        ):
            msgs = chunk.get("messages", [])
            last_count = self._print_messages_state(msgs, last_count)
            if msgs:
                final_msg = msgs[-1]
        self._printed_counts[tid] = last_count
        if final_msg is None:
            return ""
        return final_msg.content if isinstance(final_msg, AIMessage) else str(final_msg)

    def _reset_memory(self) -> None:
        """Protected: clear short-term memory for the current thread."""
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph()
        self._printed_counts[self._thread_id] = 0

    def _set_system_prompt(self, new_template: str) -> None:
        """Protected: replace the system prompt while preserving state."""
        self._system_template = new_template
        self._graph = self._build_graph()

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
        builder.add_node(call_model)
        builder.add_edge(START, "call_model")
        graph = builder.compile(checkpointer=self._checkpointer)
        return graph

    def _print_messages_state(self, messages: list[BaseMessage | dict], already_printed: int) -> int:
        try:
            total = len(messages)
            if total > already_printed:
                # Print only new messages, continue numbering
                print("MESSAGES STATE (most recent last):")
                for i in range(already_printed, total):
                    m = messages[i]
                    if isinstance(m, dict):
                        role = m.get("role", "?")
                        content = m.get("content", "")
                    else:
                        role = getattr(m, "type", getattr(m, "role", "assistant"))
                        content = getattr(m, "content", "")
                    print(f"  {i+1}. {role}: {content}")
            return total
        except Exception:
            return already_printed


'''if __name__ == "__main__":
    service = LLMService()
    print("LangChain + Ollama chat (with memory). Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.strip().lower() in {"exit", "quit"}:
            break
        try:
            # For demo purposes, context is empty; in app, pass retrieved chunks here
            ans = service.Llm_query(q, context="")
        except Exception as exc:  # noqa: BLE001
            ans = f"Error: {exc}"
        print(f"Assistant: {ans}\n")'''


