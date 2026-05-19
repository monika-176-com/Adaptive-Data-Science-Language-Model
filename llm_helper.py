from transformers import pipeline

# Load model
generator = pipeline(
    "text-generation",
    model="google/flan-t5-base"
)

def generate_llm_summary(task, results):

    prompt = f"""
    Explain the following machine learning results.

    Task:
    {task}

    Results:
    {results}

    Explain:
    - why the model was selected
    - interpretation of performance
    - suggestions for improvement
    """

    response = generator(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )

    return response[0]['generated_text']