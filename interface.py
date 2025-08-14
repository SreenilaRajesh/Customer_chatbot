import os
from typing import List, Tuple
import gradio as gr
from retriever import Retriever
from llm_module import ask_ollama
import config
from fastapi.staticfiles import StaticFiles
from llm_module2 import LLMService

def convert_src_to_html_path(src: str) -> str:
    """Return an HTTP link served from /sources without using a local 'static' folder.

    - Derive HTML name from the markdown source (basename + .html)
    - Verify it exists in config.SOURCE_FOLDER
    """
    if not src:
        return ""
    file_name = os.path.basename(src)
    base, _ext = os.path.splitext(file_name)
    base = base.split("\\")[-1]
    html_name = base + ".html"
    source_html = os.path.join(config.SOURCE_FOLDER, html_name)
    if not os.path.exists(source_html):
        return ""
    return f"{source_html}"



def respond(message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], str]:
    try:
        retriever = Retriever()
        # Use previous turn to disambiguate underspecified queries (e.g., "latest version")
        prev_question = history[-1][0] if history else ""
        combined_query = f"{message} In the context of: {prev_question}" if prev_question else message
        retrieved_chunks, retrieved_sources = retriever.retrieve_chunks(config.COLLECTION_NAME, combined_query, k=5)
        context_md = "\n\n".join(retrieved_chunks)

        # Render each dict one below the other in Markdown
        blocks = []
        for text, src in zip(retrieved_chunks, retrieved_sources):
            converted_src = convert_src_to_html_path(src) if src else ""
            if converted_src:
                link_html = f'<a href="{converted_src}" target="_blank" rel="noopener noreferrer">{converted_src}</a>'
                src_display = f"Source: {link_html}"
            else:
                src_display = "Source: (unknown)"
            block = f"**{src_display}**\n\n{text}"
            blocks.append(block)
        context_view = "\n\n---\n\n".join(blocks) if blocks else "(no context retrieved)"
        llm_service = LLMService()
        answer = llm_service.llm_query(message, context_md)
        # answer = ask_ollama(message, context_md, model_name=config.OLLAMA_MODEL)
    except Exception as exc:
        context_md = ""
        answer = f"Error: {exc}"
    history = history + [(message, answer)]
    return "", history, context_view


with gr.Blocks(title="Autodesk Chat") as demo:
    gr.Markdown("""**Autodesk Chatbot** – Ask a question""")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("**Context**")
            context_view = gr.Markdown()
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=480)
            msg = gr.Textbox(placeholder="Ask a question…", label="Message", lines=2)
            clear = gr.ClearButton([msg, chatbot])
        

    def _on_submit(user_message, chat_history):
        return respond(user_message, chat_history)

    msg.submit(
        fn=_on_submit,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, context_view],
    )

    gr.Examples(examples=[
        "How do I install the Autodesk plugin?",
        "What configuration options are available?",
        "Show me troubleshooting steps for common errors.",
    ], inputs=msg)

if __name__ == "__main__":
    # Mount SOURCE_FOLDER at /sources so links work without a local 'static' copy
    try:
        demo.app.mount("/sources", StaticFiles(directory=config.SOURCE_FOLDER), name="sources")
    except Exception:
        pass
    demo.launch(share=True)


