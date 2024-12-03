import os
from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import PromptTemplate
from tiktoken import encoding_for_model
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

# Constants
INSTRUCTIONS = """
    You are an AI assistant that generates content in two different formats depending on the query type.

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
                "sub_topics": "Subtitle 1",
                "content": "Content for subtitle 1"
            }},
            {{
                "sub_topics": "Subtitle 2",
                "content": "Content for subtitle 2"
            }}
        ],
        "summary": "A concise summary of the entire content",
        "caption": "An engaging caption for social media",
        "description": "A brief description of the content",
        "hashtags": ["#Relevant", "#Hashtags", "#ForTheContent"]
    }}

    For podcast content:
    - ONLY answer using information from the retrieved podcast documents
    - Provide detailed answers based on the podcast content
    - Do not make up or add information that isn't in the retrieved documents
    - If no podcast documents are found, respond with an error message
    Response structure for podcast content:
    {{
        "type": "podcast",
        "answer": "Direct answer from podcast content"
    }}

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Remember:
    1. For podcast queries, ONLY use information from the retrieved documents
    2. If no podcast documents are found, return: {{"type": "podcast", "answer": "No relevant podcast content found"}}
    3. Always include the "type" field: "carousel" for content generation, "podcast" for podcast queries
    4. Podcast responses should be direct answers referencing only the retrieved content
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

    def add_vector_to_index(self, files_dir, index, namespace):
        """Add documents to the vector store with a specified namespace."""
        if not namespace:
            raise ValueError("Namespace must be provided when adding documents to an index.")

        print("...Loading Data")
        documents = self.chunk_data(self.read_doc(files_dir))
        print(f"Number of chunks: {len(documents)}")
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        index = self.get_index(index)
        vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
        vector_store.add_documents(documents=documents, ids=uuids, namespace=namespace)
        print("Data loading completed!")

    def delete_index_content(self, index_name):
        """Delete all contents from an index."""
        if index_name in self.pinecone.list_indexes().names():
            index = self.pinecone.Index(index_name)
            index.delete(delete_all=True)
            print(f'Removed contents of {index_name} successfully!')
            return True
        print(f"Could not find {index_name} in the index list")
        return False

    def show_index_content(self, index_name, namespace=None):
        """Display all vectors and metadata from an index."""
        try:
            if index_name not in self.pinecone.list_indexes().names():
                print(f"Index '{index_name}' not found!")
                return

            index = self.pinecone.Index(index_name)
            query_response = index.query(
                vector=[0] * DIMENSION,
                top_k=10000,
                namespace=namespace
            )
            
            if not query_response.matches:
                print(f"No data found in index '{index_name}'" + 
                      (f" namespace '{namespace}'" if namespace else ""))
                return
            
            print(f"\nContent in index '{index_name}'" + 
                  (f" namespace '{namespace}'" if namespace else ""))
            print("-" * 50)
            
            for i, match in enumerate(query_response.matches, 1):
                print(f"\nEntry {i}:")
                print(f"ID: {match.id}")
                print(f"Score: {match.score}")
                if match.metadata:
                    print("Metadata:")
                    for key, value in match.metadata.items():
                        print(f"  {key}: {value}")
                print("-" * 30)
            
            print(f"\nTotal entries found: {len(query_response.matches)}")
            
        except Exception as e:
            print(f"Error retrieving index content: {e}")

    def get_executor(self):
        """Initialize the agent executor with necessary tools."""
        if self.agent_executor:
            return

        prompt = PromptTemplate(
            template=INSTRUCTIONS, 
            input_variables=['agent_scratchpad', 'input']
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

    def query_response(self, query):
        """Process a query and return the response in JSON format."""
        try:
            encoder = encoding_for_model(MODEL_NAME)
            try:
                query_tokens = len(encoder.encode(query))
                if query_tokens > MAX_QUERY_TOKENS:
                    query = encoder.decode(encoder.encode(query)[:MAX_QUERY_TOKENS])
            except Exception as e:
                print(f"Warning: Token counting failed: {e}")
            
            return self.agent_executor.invoke({'input': query})['output']
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "type": "error",
                "answer": "Irrelevant question"
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
            indices = self.pinecone.list_indexes().names()
            if not indices:
                print("No indices found in the vector store.")
            else:
                print("Indices in the vector store:")
                for index in indices:
                    print(f"- {index}")
            return indices
        except Exception as e:
            print(f"Error listing indices: {e}")
            return []

    def __del__(self):
        """Cleanup method to ensure proper resource handling."""
        try:
            # Clean up any open connections or resources
            if hasattr(self, 'index'):
                del self.index
            if hasattr(self, 'pinecone'):
                del self.pinecone
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Example usage
if __name__ == "__main__":
    assistant_tool = AssistantTool()
    assistant_tool.list_indices()
    # assistant_tool.add_vector_to_index('transcripts', 'master', 'podcast')
    # assistant_tool.add_vector_to_index('carousel', 'master', 'carousel')
    # print(assistant_tool.test_retrieval('What is the main topic of the podcasts?', 'podcast'))
    run = True
    while run:
        query = input('Enter: ')
        if query == 'q':
            run = False
        else:
            print(assistant_tool.query_response(query))


