from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import uuid

# Load environment variables
load_dotenv()

app = Flask(__name__)
#CORS(app)  # Enable CORS for frontend integration
CORS(app, origins=["https://easydoer.com", "https://www.easydoer.com"])

# Initialize the LLM with Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

# System message to personalize the chatbot for your portfolio
SYSTEM_MESSAGE = SystemMessage(content="""You are a helpful assistant for Prabhakar Raturi's portfolio website. 
You can answer questions about:
- His 18+ years of experience in IT and technical program management
- His expertise in digital transformation, cloud migration, and program management
- His work with organizations like Washington State, General Electric, Metrolinx, Glanbia Nutritionals
- His technical skills: ERP systems (Oracle, Workday, SAP), SailPoint, Python, AWS, Azure, PowerBI
- His certifications: PMP, CSM
- His major accomplishments in enterprise modernization, identity governance, manufacturing transformation, cloud transformation, and cybersecurity
- His location in Greater Seattle Area and contact information (phone: 425-471-2980, email: aaccela@gmail.com)

Be friendly, professional, and concise. If asked about something not related to Prabhakar's portfolio, politely redirect the conversation.""")

# Define the chatbot function
def chatbot(state: MessagesState):
    """Process messages and generate a response"""
    # Add system message if it's the first message
    messages = state["messages"]
    if len([m for m in messages if isinstance(m, SystemMessage)]) == 0:
        messages = [SYSTEM_MESSAGE] + messages
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# Build the graph
graph_builder = StateGraph(MessagesState)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Add memory to persist conversation
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests from the frontend"""
    try:
        data = request.json
        user_message = data.get('message', '')
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Configure session for memory persistence
        config = {"configurable": {"thread_id": session_id}}
        
        # Invoke the chatbot
        result = graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config
        )
        
        # Get the AI's response
        ai_response = result["messages"][-1].content
        
        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request'}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)