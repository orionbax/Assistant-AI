import langchain
# import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import os
from uuid import uuid4

from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from pprint import pprint
from langchain.prompts import PromptTemplate

load_dotenv()

transcripts_dir = 'transcripts'
documents_dir = 'documents'
corousels_dir = 'corousels_txt'
book_drafts_dir = ''

api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_DEFAULT_API_KEY_ATHEM")
pinecone_api_key = os.getenv("PINECONE_API_KEY")


INSTRUCTIONS = """
    You should not answer any inquiries out of your scope.
    You are here to assist with generating new content for the provided topic, assume the content is for linked in.
    The new content should somewhat reflect styles of the previous posts
    Adhere to the following principles and guidelines when you proceed

    -Tone: Conversational, relatable, empathetic, and human. Avoid robotic or overly technical language.
    -Voice: Authoritative yet approachable, focused on problem-solving and building connections.
    -Avoid: Jargon, emojis, and gimmicks.
    -Use a clear structure: Start with a hook, and end with actionable insights.
    -decent amount of paragraphs, bullet points, and headings for readability.
    -Always provide with a capturing caption.
    -Balance storytelling with actionable information.
    -Focus on relatable, audience-centered content.
    -Use simple, clear, and direct language. Prioritize empathy and relatability.
    -Avoid: Buzzwords, clichés, and overused phrases.   
    -Use concise, impactful phrasing.
    -Keep sentences short for intros and conclusions.
    -Core Values: Clear communication that addresses audience needs and pain points.
    -Positioning: Emphasize leadership, versatility, and strong audience connections.
    -Do's: Address audience pain points and offer solutions.
    -Don'ts: Avoid redundancy or filler content.
    -Follow a storytelling framework: Hook - Problem - Solution - Call to Action.
    -Creative Freedom: You can experiment with tones/styles if relatable.
    -Defined Parameters: Ensure tone is empathetic, simple, and authentic. Stay audience-focused without being overly promotional.

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Respond accordingly based on the above context.
"""

# """

    # JSON Response Format:

    # "summary": "A concise summary of the response.",
    # "title": What should the title of the post be,
    # "content": the content goes in here,
    # "caption": caption of the post,
    # "hashtags" : "the relevant hashtags to use"
    # "resources": "The basis of your answer and the documents you've uses as a reference"
    
    # If no relevant content is found in the context, respond with: 
    #     "error": "No relevant content found in the LinkedIn posts.".

#   Your response must always be in a json form as such whenever the context is above content generation!

#     {
#         'content_for': linked_in (this is case sensitive so keep like this and dotn change it)
#         "Title": "Put the tile here"
#         "content": "CONTENT GOES IN HERE, NEEDS TO BE AS DETAILED AS POSSIBLE",
#         "caption": "Put the caption here",
#         "hashtags": "#Hashtag1 #hashtag2 etc",
#         "Summary": If you think is necessary,
#         "resources" The basis of your answer, could be file name or could be anything just make it clear wether it was from your memory or outside source! 
#     }
# """


INSTRUCTION_TWO = """
    You are a helpful assistant designed to generate new content for the user based on their existing LinkedIn posts stored in Pinecone. 
    You must strictly adhere to the guidelines below and only provide responses derived from the provided context. 

    Guidelines:
    - Your response must be in JSON format, following the example structure below.

    JSON Response Format:
   
        "summary": "A concise summary of the response.",
        "title": What should the title of the post be,
        "content": the content goes in here,
        "caption": caption of the post,
        "hashtags" : "the relevant hashtags to use"
        "resources": "The basis of your answer and the documents you've uses as a reference"
        
        If no relevant content is found in the context, respond with: 
            "error": "No relevant content found in the LinkedIn posts.".

    Tone and Voice:
    - Tone: Conversational, relatable, empathetic, and human. Avoid robotic or overly technical language.
    - Voice: Authoritative yet approachable, focused on problem-solving and building connections.

    Structure:
    - Use a clear structure: Start with a hook, and end with actionable insights.
    - Avoid redundancy, jargon, buzzwords, or unnecessary filler words.

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Respond with a JSON-formatted response based on the user's query and the relevant LinkedIn posts provided.


"""


class AssistantTool:
    def __init__(self):
        self.pinecone = Pinecone(
            api_key=pinecone_api_key
        )
        self.index_name = "corousels"
        self.embeddings = OpenAIEmbeddings(api_key=api_key,model='text-embedding-3-large')

        self.index = self.get_index(self.index_name)
        self.vector_store= PineconeVectorStore(index=self.index, embedding=self.embeddings)
        self.llm = ChatOpenAI(
            api_key=api_key, 
            model='gpt-4', 
            temperature=0.7,
            max_tokens=None,
        )
        self.agent_executor = None 
        self.get_executor()


    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50, namespace=""):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(docs)
        for i, d in enumerate(docs):
            d.metadata['namespace'] = namespace
        return docs
    

    def read_doc(self, directory):
        file_loader = PyPDFDirectoryLoader(directory)
        document = file_loader.load()

        return document
    
    def add_vector_to_index(self, files_dir, namespace):
        print("...Loading Data")
        documents = self.chunk_data(self.read_doc(files_dir), namespace=namespace)
        print(len(documents))
        # pprint(documents)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids) #  use the documents id instead
        print("Done!")    

    def get_index(self, index_name):
        if index_name not in self.pinecone.list_indexes().names():
            self.pinecone.create_index(
                name=index_name, 
                dimension=3072, 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        index = self.pinecone.Index(index_name)
        # print("Index was created successfully")
        return index 
    
    def get_executor(self):
        if self.agent_executor:
            return 
        
        prompt = PromptTemplate(template=INSTRUCTIONS, input_variables=['agent_scratchpad', 'input'])
        corousels_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'corousels'}},)
        podcast_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'podcast'}},)

        podcast_tool = create_retriever_tool(
            podcast_retriever,
            "podcast_summary",
            "If you find relevant information about Podcast use this tool to give a summary"
        )


        corousels_tool = create_retriever_tool(
            corousels_retriever,
            "corousels_generator",
            f""" Use this tool if asked to generate new content
            """

            # there should be nothing else other that the json in your response
            #     {
            #     'content_for': linked_in (this is case sensitive so keep like this and dotn change it)
            #     "title": "Put the tile here"
            #     "content": [{'title': 'paragraph'}] put as many in the list as you see fit
            #     "caption": "Put the caption here",
            #     "hashtags": "#Hashtag1 #hashtag2 etc",
            #     "summary": If you think is necessary,
            #     "resources" The basis of your answer and where you got it from, very important
            #     }
        )
        podcast_tool = create_retriever_tool(
            None,
            "podcast_summary",
            'If the questions is out of the scope: simply answer with "Out of scope! I cant give you an answer to that'
        )

        tools = [ corousels_tool]
        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)
        # return self.agent_executor
    
    def query_response(self, query):

        result = self.agent_executor.invoke({'input': query})
        return result['output']



    

assistant_tool = AssistantTool()
# print(assistant_tool.query_response("Need a new contenet about being Confident in yoru actions."))
print(assistant_tool.query_response("Need a content about Building sustainable funding strategy"))

# main(assistant_tool.vector_store)