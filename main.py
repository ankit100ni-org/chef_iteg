# main.py
from transformers import pipeline

def load_qa_model(model_name="distilbert-base-cased-distilled-squad"):
    """Load a question-answering model from Hugging Face."""
    print(f"Loading model: {model_name}")
    model = pipeline('question-answering', model=model_name)
    return model

def get_answer(model, question, context):
    """Get an answer to a question based on the context."""
    response = model(question=question, context=context)
    return response['answer']

if __name__ == "__main__":
    # Define the question and context
    question = "Who is the president of the United States?"
    context = "Joe Biden is the president of the United States."

    # Load the QA model from Hugging Face
    model = load_qa_model()
    
    # Get the answer
    answer = get_answer(model, question, context)
    print("Answer: ", answer)
