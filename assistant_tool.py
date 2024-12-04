import os
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate
from tiktoken import encoding_for_model
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_community.document_loaders import PyPDFDirectoryLoader
from typing import Dict
import regex
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
# Load environment variables
load_dotenv()

# Constants
INSTRUCTIONS = """
    You are a specialized AI assistant that MUST ALWAYS respond in valid JSON format.
    You are focused ONLY on three specific types of content:
    1. Social media content generation (specifically carousels)
    2. Podcast content retrieval and information
    3. Previous conversation retrieval and information

    CRITICAL RESPONSE RULES:
    - You MUST ALWAYS respond with properly formatted JSON
    - NO additional text before or after the JSON
    - NO explanations outside the JSON structure
    - Make maximum TWO tool calls per query
    - After getting tool responses, format your final answer immediately as JSON
    - NEVER make additional tool calls after receiving the initial responses
    - If you are asked to modify the content, DO NOT make any more calls.

    CONTENT GUIDELINES:
    For social media content (carousel type):
    - Maintain a professional yet conversational tone
    - Focus on value and actionable insights
    - Keep content detailed and engaging
    - Use clear structure and formatting
    - Avoid jargon and buzzwords
    - Try and reflect the style of the previous posts from the provided documents

    For podcast content:
    - BASE your answer on information from the retrieved podcast documents ONLY
    - If no podcast documents are found, respond with the no-content JSON response
    - Keep responses focused and direct
    - Include relevant quotes or specific references when available

    RESPONSE FORMATS:

    1. For social media carousel content:
    {{
        "type": "carousel",
        "topic": "Main topic title",
        "paragraph": "Introduction paragraph that sets the context",
        "body": [
            {{
                "sub_topics": "Subtitle",
                "content": "Content for this subtitle"
            }}
        ],
        "summary": "A concise summary of the entire content",
        "caption": "An engaging caption for social media",
        "description": "A brief description of the content",
        "hashtags": ["#Relevant", "#Hashtags", "#ForTheContent"]
    }}

    2. For podcast content:
    {{
        "type": "podcast",
        "answer": "Direct answer from podcast content"
    }}

    3. For conversation history:
    {{
        "type": "error",
        "answer": "Response about conversation history. If no history exists, explain that there are no previous conversations."
    }}

    4. For errors or out-of-scope queries:
    {{
        "type": "error",
        "answer": "I am a specialized assistant focused only on social media carousel content, podcast information, and conversation history. I cannot help with [specific topic]. Please ask questions related to these topics."
    }}

    5. For content modification requests:
    - Use the same JSON structure as the original content type
    - Do not make new tool calls
    - Modify the existing content while maintaining proper JSON format
    - Preserve the original style and tone while making requested changes

    STRICT RULES:
    1. ALWAYS respond with ONE of the above JSON formats
    2. NEVER include any text outside the JSON structure
    3. For podcast queries, use ONLY information from the retrieved documents
    4. If no podcast documents found: {{"type": "podcast", "answer": "No relevant podcast content found"}}
    5. ALWAYS include the "type" field
    6. DO NOT attempt to answer questions outside the three focus areas
    7. Maintain consistent formatting and structure in responses
    8. For carousel content, always include all required fields
    9. Keep responses focused and relevant to the query
    10. Preserve professional tone while remaining engaging

    Previous conversation history:
    {conversation_history}

    Context so far:
    {agent_scratchpad}

    User query:
    {input}
"""

DIMENSION = 3072
CHUNK_SIZE = 800 
CHUNK_OVERLAP = 50
MAX_QUERY_TOKENS = 6000
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-large"

class AssistantTool:
    def __init__(self, debug=False):
        """Initialize the AssistantTool with necessary configurations."""
        self.debug = debug
        self._validate_env_vars()
        self._initialize_components()
        self.agent_executor = None
        self.get_executor()
        self.conversation_histories: Dict[str, list] = {}

    def _validate_env_vars(self):
        """Validate required environment variables are present."""
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _initialize_components(self):
        """Initialize Pinecone, embeddings, and other components."""
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        # Initialize two separate indices
        self.carousel_index_name = 'carousel'
        self.podcast_index_name = 'podcast'
        
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=EMBEDDING_MODEL
        )
        
        # Create/get both indices
        self.carousel_index = self._get_index(self.carousel_index_name)
        self.podcast_index = self._get_index(self.podcast_index_name)
        
        # Create vector stores for both indices
        self.carousel_vector_store = PineconeVectorStore(
            index=self.carousel_index, 
            embedding=self.embeddings
        )
        self.podcast_vector_store = PineconeVectorStore(
            index=self.podcast_index, 
            embedding=self.embeddings
        )
        
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=MODEL_NAME,
            temperature=0.7
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_index(self, index_name):
        """Get or create a Pinecone index with retries."""
        try:
            if index_name not in self.pinecone.list_indexes().names():
                self.pinecone.create_index(
                    name=index_name,
                    dimension=DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
            return self.pinecone.Index(index_name)
        except Exception as e:
            raise RuntimeError(f"Failed to get or create Pinecone index: {str(e)}")

    def chunk_data(self, docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def read_doc(self, directory):
        """Load documents from a directory."""
        return PyPDFDirectoryLoader(directory).load()

    def add_vector_to_index(self, files_dir, content_type):
        """Add documents to the vector store based on content type."""
        if content_type not in ['carousel', 'podcast']:
            raise ValueError("content_type must be either 'carousel' or 'podcast'")
        
        documents = self.chunk_data(self.read_doc(files_dir))
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        if content_type == 'carousel':
            self.carousel_vector_store.add_documents(documents=documents, ids=uuids)
        else:  # podcast
            self.podcast_vector_store.add_documents(documents=documents, ids=uuids)

    def delete_index_content(self, index_name):
        """Delete all contents from an index."""
        if index_name in self.pinecone.list_indexes().names():
            index = self.pinecone.Index(index_name)
            index.delete(delete_all=True)
            return True
        return False

    def show_index_content(self, index_name, namespace=None):
        """Display all vectors and metadata from an index."""
        try:
            if index_name not in self.pinecone.list_indexes().names():
                return

            index = self.pinecone.Index(index_name)
            query_response = index.query(
                vector=[0] * DIMENSION,
                top_k=10000,
                namespace=namespace
            )
            
            if not query_response.matches:
                return
            
            return query_response.matches
            
        except Exception as e:
            raise RuntimeError(f"Error retrieving index content: {e}")

    def format_conversation_history(self, user_id: str) -> str:
        """Format conversation history for the prompt for a specific user."""
        if user_id not in self.conversation_histories or not self.conversation_histories[user_id]:
            return "No previous conversation."
        
        formatted_history = []
        for i, exchange in enumerate(self.conversation_histories[user_id][-3:], 1):  # Only use last 3 exchanges
            formatted_history.extend([
                f"Exchange {i}:",
                f"User: {exchange['user_input']}",
                f"Assistant: {exchange['response']}",
                ""
            ])
        return "\n".join(formatted_history)

    def get_executor(self):
        """Initialize the agent executor with necessary tools."""
        if self.agent_executor:
            return

        prompt = PromptTemplate(
            template=INSTRUCTIONS, 
            input_variables=['agent_scratchpad', 'input', 'conversation_history']
        )

        # Use separate vector stores for each content type
        carousel_retriever = self.carousel_vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        )
        podcast_retriever = self.podcast_vector_store.as_retriever(
            search_kwargs={
                "k": 4
            }
        )

        # Create tools with the separate retrievers
        tools = [
            create_retriever_tool(
                retriever=carousel_retriever,
                name="social_media_content",
                description="Use this tool for social media content generation, LinkedIn posts, and content strategy queries. Make only one call to this tool."
            ),
            create_retriever_tool(
                retriever=podcast_retriever,
                name="podcast_content",
                description="Use this tool for podcast-related queries, summaries, and insights from podcast episodes. Make only one call to this tool."
            )
        ]

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            verbose=self.debug,
            tools=tools, 
            max_iterations=3,  # Reduced from 25 to 3
            early_stopping_method="force"  # Force stop after max_iterations
        )

    def add_to_history(self, user_id: str, user_input: str, response: str):
        """Add a user query and response to the conversation history for a specific user."""
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
            
        self.conversation_histories[user_id].append({
            "user_input": user_input,
            "response": response
        })

    def get_conversation_history(self, user_id: str) -> list:
        """Retrieve the conversation history for a specific user."""
        return self.conversation_histories.get(user_id, [])

    def generate_user_id(self) -> str:
        """Generate a unique user ID."""
        return str(uuid4())

    def clean_ai_response(self, response):
        # Recursive pattern for JSON-like structures
        json_like_pattern = r'{(?:[^{}]|(?R))*}'  # Recursive matching of nested braces
        
        # Use the regex library to match
        match = regex.search(json_like_pattern, response)
        if match:
            json_text = match.group(0)
            try:
                # Validate the extracted text as JSON
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass

        return "{}"  # Return empty JSON if no valid structure found

    def query_response(self, query: str, user_id: str):
        """Process a query and return the response in JSON format."""
        try:
            encoder = encoding_for_model(MODEL_NAME)
            try:
                query_tokens = len(encoder.encode(query))
                if query_tokens > MAX_QUERY_TOKENS:
                    query = encoder.decode(encoder.encode(query)[:MAX_QUERY_TOKENS])
            except Exception:
                pass
            
            # Add a timeout to prevent infinite loops
            with ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.agent_executor.invoke,
                    {
                        'input': query,
                        'conversation_history': self.format_conversation_history(user_id)
                    }
                )
                try:
                    response = future.result(timeout=180)  # 30 second timeout
                    response = response['output']
                    response = self.clean_ai_response(response)
                    # print(response)
                    # Extract JSON from response
                except TimeoutError:
                    response = {
                        "type": "error",
                        "answer": "Request timed out. Please try again with a more specific query."
                    }
            
            self.add_to_history(user_id, query, response)
            return response
        except Exception as e:
            return {
                "type": "error",
                "answer": f"Error processing query: {str(e)}"
            }

    def test_retrieval(self, query, content_type):
        """Test retrieval functionality for podcast or carousel content."""
        try:
            if content_type not in ['podcast', 'carousel']:
                raise ValueError("content_type must be either 'podcast' or 'carousel'")

            # Use the appropriate vector store based on content type
            vector_store = (
                self.podcast_vector_store if content_type == 'podcast' 
                else self.carousel_vector_store
            )
            
            retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 4
                }
            )

            docs = retriever.invoke(query)
            
            results = {
                "query": query,
                "content_type": content_type,
                "num_docs_found": len(docs),
                "documents": []
            }
            
            for i, doc in enumerate(docs, 1):
                results["documents"].append({
                    "document_number": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return results

        except Exception as e:
            return {
                "error": str(e),
                "query": query,
                "content_type": content_type,
                "num_docs_found": 0,
                "documents": []
            }

    def list_indices(self):
        """List all indices in the Pinecone vector store."""
        try:
            indices = self.pinecone.list_indexes()
            indices = [index.name for index in indices if index.name in ['carousel', 'podcast']]
            return indices
        except Exception as e:
            print(f"Error listing indices: {e}")
            return []

    def __del__(self):
        """Cleanup method to ensure proper resource handling."""
        try:
            if hasattr(self, 'index'):
                del self.index
            if hasattr(self, 'pinecone'):
                del self.pinecone
        except Exception:
            pass

# # Example usage
if __name__ == "__main__":
    assistant_tool = AssistantTool()
    # print(assistant_tool.list_indices())
#     # print(assistant_tool._get_index('master'))
#     assistant_tool.add_vector_to_index('transcripts', 'transcripts')
    # assistant_tool.add_vector_to_index('corousels_txt', 'carousel')
#     # print(assistant_tool.test_retrieval('What is the main topic of the podcasts?', 'podcast'))
    run = True
    test_user_id = input("enter your key: ")  # Example user ID for testing
    while run:
        query = input('Enter: ')
        if query == 'q':
            run = False
        else:
            print(assistant_tool.query_response(query, test_user_id))



