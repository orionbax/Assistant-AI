from flask import Flask, request, render_template, jsonify
# import assistant_tool
from assistant_tool import AssistantTool
import os 

# Create a singleton instance outside of the Flask app
assistant_tool = AssistantTool()

# Initialize Flask app
app = Flask(__name__)


# Sample data dictionary
# sample_data = {
#     "topic": "Building a Sustainable Funding Strategy",
#     "paragraph": "Securing funding for a startup is a challenging task that requires a well-thought-out strategy. This strategy should be flexible enough to evolve with your product and market growth. One must remember that in this journey, funding is not a one-off event but an ongoing process.",
#     "body": [
#         {
#             "sub_topics": "Staged Funding",
#             "content": "Instead of seeking a large capital injection at once, it would be wise to break your funding strategy into stages. Each stage should align with key milestones in your company's development."
#         },
#         {
#             "sub_topics": "Identify Key Milestones", 
#             "content": "Identify specific milestones like user adoption, partnerships with artists/platforms, or technology enhancements. Use these milestones to demonstrate continuous progress to investors."
#         },
#         {
#             "sub_topics": "Building Investor Trust",
#             "content": "Raise capital progressively and show how each round drives growth in your solution. Clear, measurable progress will build long-term trust with investors."
#         },
#         {
#             "sub_topics": "Sustainable Growth",
#             "content": "The focus should be on building a solid foundation and growing steadily rather than scaling fast with massive capital. Remember, great success takes time!"
#         },
#         {
#             "sub_topics": "Leverage Each Investment",
#             "content": "Make the most of each investment, raise what you need to hit the next significant milestone. This approach builds investor confidence for future rounds."
#         }
#     ],
#     "summary": "A sustainable funding strategy is about planning for staged funding, identifying key milestones, building investor trust, focusing on sustainable growth, and leveraging each investment effectively. It's a journey that evolves with your product and market growth, and it requires patience and strategic planning.",
#     "caption": "Building a Sustainable Funding Strategy: A Journey Not a Destination",
#     "description": "This article discusses how to build a sustainable funding strategy for startups, focusing on planning for staged funding, identifying key milestones, building investor trust, focusing on sustainable growth, and leveraging each investment effectively.",
#     "hashtags": ["#StartupFunding", "#InvestmentStrategy", "#SustainableGrowth", "#InvestorRelations", "#Funding"]
# }

@app.route('/start', methods=['GET'])
def start():
    """Return the sample data dictionary."""
    return jsonify({})


@app.route('/')
def chat_interface():
    """Render the chat interface."""
    return render_template('cursor_test.html')

@app.route('/answer', methods=['POST'])
def answer():
    """Process user query and return response."""
    try:
        # Get user query from request
        data = request.json
        user_query = data.get('message')
        
        if not user_query:
            return jsonify({
                "type": "podcast",
                "answer": "No query provided"
            }), 400
            
        # Use the AssistantTool to get response
        print(f'user query: {user_query}')
        response = assistant_tool.query_response(user_query)   
        return response
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({
            "type": "podcast",
            "answer": "An error occurred while processing your request"
        }), 500

if __name__ == "__main__":
    app.run()

# what did i do today: