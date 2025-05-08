from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# Model configuration
model_id = "Salesforce/xLAM-2-1b-fc-r"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    return_full_text=False
)

# Create LangChain pipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt template
template = """You are a helpful AI assistant. Respond to the user's Question:

Question: {question}

Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])
chain = prompt | llm

# Chatbot interface
print("Chatbot: Hello! I'm your AI assistant. Type 'exit' to end the conversation.")
while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye! Have a great day!")
        break
    
    try:
        response = chain.invoke({"question": user_input})
        print(f"Chatbot: {response}")
    except Exception as e:
        print(f"Chatbot: Sorry, I encountered an error. Could you try again? ({str(e)})")