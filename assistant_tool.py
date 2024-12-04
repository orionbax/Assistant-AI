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
# Load environment variables
load_dotenv()

# Constants
INSTRUCTIONS = """
    You are a specialized AI assistant focused ONLY on two specific types of content:
    1. Social media content generation (specifically carousels)
    2. Podcast content retrieval and information

    If a query is outside these two domains, politely explain that you are a specialized assistant 
    focused only on social media carousel content and podcast information, and cannot help with other topics.

    For social media content (carousel type):
    - Maintain a professional yet conversational tone
    - Focus on value and actionable insights
    - Keep content detailed and engaging
    - Use clear structure and formatting
    - Avoid jargon and buzzwords
    - Ensure content reflects previous post styles
    Response structure for carousel content:
    {{
        "type": "carousel",
        "topic": "Main topic title",
        "paragraph": "Introduction paragraph that sets the context",
        "body": [
            {{
                "sub_topics": "Subtitle",
                "content": "Content for this subtitle"
            }}
            // Additional sub-topics as needed...
        ],
        "summary": "A concise summary of the entire content",
        "caption": "An engaging caption for social media",
        "description": "A brief description of the content",
        "hashtags": ["#Relevant", "#Hashtags", "#ForTheContent"]
    }}

    For podcast content:
    - BASE your answer on information from the retrieved podcast documents
    - Do not make up or add information that isn't in the retrieved documents
    - If no podcast documents are found, respond with an error message
    Response structure for podcast content:
    {{
        "type": "podcast",
        "answer": "Direct answer from podcast content"
    }}

    For out-of-scope queries:
    {{
        "type": "error",
        "answer": "I am a specialized assistant focused only on social media carousel content and podcast information. I cannot help with [specific topic]. Please ask questions related to social media carousel content creation or podcast information."
    }}

    Previous conversation history:
    {conversation_history}

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Remember:
    1. For podcast queries, ONLY use information from the retrieved documents
    2. If no podcast documents are found, return: {{"type": "podcast", "answer": "No relevant podcast content found"}}
    3. Always include the "type" field: "carousel" for content generation, "podcast" for podcast queries, "error" for out-of-scope queries
    4. Podcast responses should be direct answers referencing only the retrieved content
    5. Do not attempt to answer questions outside of social media carousel content and podcast information
"""

DIMENSION = 3072
CHUNK_SIZE = 800 
CHUNK_OVERLAP = 50
MAX_QUERY_TOKENS = 6000
MODEL_NAME = "gpt-4"
EMBEDDING_MODEL = "text-embedding-3-large"

class AssistantTool:
    def __init__(self):
        """Initialize the AssistantTool with necessary configurations."""
        self._validate_env_vars()
        self._initialize_components()
        self.agent_executor = None
        self.get_executor()
        self.conversation_history = []  # Initialize conversation history

    def _validate_env_vars(self):
        """Validate required environment variables are present."""
        required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _initialize_components(self):
        """Initialize Pinecone, embeddings, and other components."""
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = 'master'
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=EMBEDDING_MODEL
        )
        self.index = self._get_index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index, 
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

    def add_vector_to_index(self, files_dir, index_name, namespace=""):
        """Add documents to the vector store."""
        documents = self.chunk_data(self.read_doc(files_dir))
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        index = self._get_index(index_name)
        vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
        vector_store.add_documents(documents=documents, ids=uuids)

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

    def format_conversation_history(self):
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted_history = []
        for i, exchange in enumerate(self.conversation_history[-3:], 1):  # Only use last 3 exchanges
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

        # Updated retrievers without include_metadata argument
        corousels_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 4,
                "namespace": "carousel"
            }
        )
        podcast_retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 4,
                "namespace": "podcast"
            }
        )

        # Create tools
        tools = [
            create_retriever_tool(
                retriever=corousels_retriever,
                name="social_media_content",
                description="Use this tool for social media content generation, LinkedIn posts, and content strategy queries"
            ),
            create_retriever_tool(
                retriever=podcast_retriever,
                name="podcast_content",
                description="Use this tool for podcast-related queries, summaries, and insights from podcast episodes"
            )
        ]

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            verbose=True, 
            tools=tools, 
            max_iterations=25
        )

    def add_to_history(self, user_input, response):
        """Add a user query and response to the conversation history."""
        self.conversation_history.append({
            "user_input": user_input,
            "response": response
        })

    def get_conversation_history(self):
        """Retrieve the conversation history."""
        return self.conversation_history

    def query_response(self, query):
        """Process a query and return the response in JSON format."""
        try:
            encoder = encoding_for_model(MODEL_NAME)
            try:
                query_tokens = len(encoder.encode(query))
                if query_tokens > MAX_QUERY_TOKENS:
                    query = encoder.decode(encoder.encode(query)[:MAX_QUERY_TOKENS])
            except Exception as e:
                pass
            
            # Include conversation history in the context
            conversation_history = self.format_conversation_history()
            response = self.agent_executor.invoke({
                'input': query,
                'conversation_history': conversation_history
            })['output']
            
            self.add_to_history(query, response)  # Add to history
            print(response)
            return response
        except Exception as e:
            return {
                "type": "error",
                "answer": f"Error processing query: {str(e)}"
            }

    def test_retrieval(self, query, content_type):
        """
        Test retrieval functionality for podcast or carousel content.
        
        Args:
            query (str): The search query
            content_type (str): Either 'podcast' or 'carousel'
            
        Returns:
            dict: Retrieved documents and their metadata
        """
        try:
            # Validate content type
            if content_type not in ['podcast', 'carousel']:
                raise ValueError("content_type must be either 'podcast' or 'carousel'")

            # Updated retriever without type filter
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "k": 4,
                    "namespace": content_type,
                }
            )

            # Get documents
            docs = retriever.invoke(query)
            
            # Format results
            results = {
                "query": query,
                "content_type": content_type,
                "num_docs_found": len(docs),
                "documents": []
            }
            
            # Add each document's content and metadata
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
            indices = [index.name for index in indices]
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
#     # print(assistant_tool._get_index('master'))
#     assistant_tool.add_vector_to_index('transcripts', 'transcripts')
#     # assistant_tool.add_vector_to_index('carousel', 'master', 'carousel')
#     # print(assistant_tool.test_retrieval('What is the main topic of the podcasts?', 'podcast'))
    run = True
    while run:
        query = input('Enter: ')
        if query == 'q':
            run = False
        else:
            print(assistant_tool.query_response(query))



