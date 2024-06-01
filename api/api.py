import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
# Append the path for the RAG module
from RAG import AIAgent, RAGSystem 
# Initialize the AI Agent
ai_agent = AIAgent()
# Initialize the RAGSystem with the existing collection
rag_system = RAGSystem(ai_agent=ai_agent, num_retrieved_docs=4)
# Initialize FastAPI app
# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/query", methods=["POST"])
def generate_answer():
    try:
        # Get the prompt from the request JSON
        data = request.json
        prompt = data.get("prompt", "")

        # Query the RAG system with the provided prompt
        res = rag_system.query(prompt)

        # Return the response
        return jsonify({"response": res})
    except Exception as e:
        # Return an error response
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)
