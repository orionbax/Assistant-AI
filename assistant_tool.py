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

# Load environment variables
load_dotenv()

# Constants
INSTRUCTIONS = """
    If you're asked about content generation, LinkedIn, or things that have to do with social media, keep the following in mind:
    - You should not answer any inquiries out of your scope.
    - You are here to assist with generating new content for the provided topic, assume the content is for LinkedIn.
    - The new content should somewhat reflect styles of the previous posts.
    - Adhere to the following principles and guidelines when you proceed:
      - Tone: Conversational, relatable, empathetic, and human. Avoid robotic or overly technical language.
      - Voice: Authoritative yet approachable, focused on problem-solving and building connections.
      - Avoid: Jargon, emojis, and gimmicks.
      - Use a clear structure: Start with a hook, and end with actionable insights.
      - Decent amount of paragraphs, bullet points, and headings for readability.
      - Always provide a capturing caption.
      - Balance storytelling with actionable information.
      - Focus on relatable, audience-centered content.
      - Use simple, clear, and direct language. Prioritize empathy and relatability.
      - Avoid: Buzzwords, clichés, and overused phrases.
      - Use concise, impactful phrasing.
      - Keep sentences short for intros and conclusions.
      - Core Values: Clear communication that addresses audience needs and pain points.
      - Positioning: Emphasize leadership, versatility, and strong audience connections.
      - Do's: Address audience pain points and offer solutions.
      - Don'ts: Avoid redundancy or filler content.
      - Follow a storytelling framework: Hook - Problem - Solution - Call to Action.
      - Creative Freedom: You can experiment with tones/styles if relatable.
      - Defined Parameters: Ensure tone is empathetic, simple, and authentic. Stay audience-focused without being overly promotional.

    For podcast-related queries:
    - Summarize the podcast content succinctly.
    - Highlight key points and insights from the podcast.
    - Provide a brief overview of the podcast's theme and any notable guests or discussions.
    - Use a similar tone and style as above, focusing on clarity and engagement.

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Respond accordingly based on the above context.
"""

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
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_DEFAULT_API_KEY_ATHEM"))
        self.index_name = "corousels2"
        self.embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            model='text-embedding-3-large'
        )
        self.index = self.get_index(self.index_name)
        self.vector_store = PineconeVectorStore(
            index=self.index, 
            embedding=self.embeddings
        )
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model='gpt-4',
            temperature=0.7,
            max_tokens=None
        )

    def get_index(self, index_name):
        """Get or create a Pinecone index."""
        if index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=index_name,
                dimension=3072,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        return self.pinecone.Index(index_name)

    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50):
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
        vector_store.add_documents(documents=documents, ids=uuids)
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
                vector=[0] * 3072,
                top_k=10000,
                include_metadata=True,
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

        # Create retrievers with appropriate filters
        corousels_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4, 'filter': {'namespace': 'corousels'}}
        )
        podcast_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4, 'filter': {'namespace': 'podcast'}}
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
        """Process a query and return the response."""
        try:
            result = self.agent_executor.invoke({'input': query})
            return result['output']
        except Exception as e:
            print(f"Error processing query: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."

# Example usage
if __name__ == "__main__":
    assistant_tool = AssistantTool()
    # assistant_tool.show_index_content('corousels2')
    main_index_name = 'master'
    assistant_tool.add_vector_to_index('corousels_txt', main_index_name, 'corousel')
    assistant_tool.add_vector_to_index('transcripts', main_index_name, 'podcast')
