# app/main.py
import logging
from app.graph import build_graph
from app.state import GraphState

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Company Policy Chatbot...")
    graph = build_graph()
    
    # Initialize empty state
    state = {
        "original_question": "",
        "question": "",
        "chat_history": [],
        "query_type": "",
        "documents": [],
        "context": "",
        "answer": "",
        "validation_result": False,
        "feedback": {}
    }
    
    print("\n--- Company Policy Assistant ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Update state with new question
        state["original_question"] = user_input
        
        try:
            # Run the graph
            result = graph.invoke(state)
            
            # Print the answer
            print(f"\nAssistant: {result['answer']}\n")
            
            # Update the persistent state with new history and for the next loop
            state["chat_history"] = result["chat_history"]
            
        except Exception as e:
            logger.error(f"Graph execution error: {e}")
            print("Assistant: I encountered a critical error. Please try again.")

if __name__ == "__main__":
    main()