from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

# Load the pre-trained DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Function to get the answer from the pre-trained DistilBERT model
def get_answer(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax() + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start_index:answer_end_index])
    )
    return answer.strip()

# Test with a few sample questions and contexts
def test_model():
    questions = [
        "What is diabetes?",
        "What is the capital of France?",
        "How do I treat a headache?",
        "Who is the president of the United States?",
        "What is cholesterol?"
    ]

    contexts = [
        "Diabetes is a long-lasting health condition that affects how your body turns food into energy.",
        "Paris is the capital of France.",
        "A headache can be treated by resting, taking pain relievers, and staying hydrated.",
        "The president of the United States is Joe Biden.",
        "Cholesterol is a substance found in your blood that your body needs to build healthy cells."
    ]

    for question, context in zip(questions, contexts):
        answer = get_answer(question, context)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print()

# Run the test
test_model()
