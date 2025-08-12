import ollama
from datetime import datetime
import config
from retriever import Retriever



def ask_ollama(query, context, model_name):
    prompt = f"""
        You are a question-answering assistant.
        You are given the following context extracted from documents:

        {context}

        Answer the question below *only* using the context.
        If the answer is not present in the context, reply exactly:
        "The answer is not available in the provided context."

        Question: {query}
    """
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']


'''if __name__ == "__main__":

    print("Chat with Autodesk Chatbot (type 'exit' to quit)")
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        retriever = Retriever()
        retrieved = retriever.retrive_chunks(config.COLLECTION_NAME, query, 5)
        context = "\n\n".join(retrieved)
        #print(context)
        answer = ask_ollama(query, context, config.OLLAMA_MODEL)
        print(f"Response: {answer}")'''