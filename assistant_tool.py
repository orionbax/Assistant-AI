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
pinecone_api_key = os.getenv("PINECONE_DEFAULT_API_KEY_ATHEM")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
COROUSEL_INSTRUCTIONS = """
    1, You are here to assist with generating new content for the provided query, assume the content is for linked in.
    2, The new content should somewhat reflect styles of the previous posts, If available they'll be available to you
    3, wrap the contents of your response with the appropriate html tag so it can be displayed nicely on webpages,
    4, Adhere to the following principles and guidelines when you proceed

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


    query:
    {page_content}

    Respond accordingly based on the above query.
"""
PODCAST_INSTRUCTIONS="""
    -Respond accordingly based on the below query
    
    query:
    {page_content}
    
"""

INSTRUCTIONS = """
    You should not answer any inquiries out of your scope.
    You are here to assist with generating new content for the provided topic, assume the content is for linked in.
    The new content should somewhat reflect styles of the previous posts
    Adhere to the following principles and guidelines when you proceed
    wrap the contents of your response with the appropriate html tag,

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

INSUTRUCTION_TWO = ""
class AssistantTool:
    def __init__(self):
        self.indices = [
            {"name": 'corousels',
             'description': "Use this for social media or content generation related queries",
             "document_prompt": COROUSEL_INSTRUCTIONS
             },

             {  "name": 'transcripts',
                'description': "For any relevant queries to Podcasts use this tool",
                "document_prompt": PODCAST_INSTRUCTIONS
             }
        ]
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
    
    def delete_index_content(self, index_name):
        if index_name in self.pinecone.list_indexes().names():
            index = self.pinecone.Index(index_name)
            index.delete(delete_all=True)
            print(f'Removed contents of {index_name} successfully!')
            return True
        print(f"Could not find {index_name} in the index list")

    def chunk_data(self, docs, chunk_size=800, chunk_overlap=50, namespace=""):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs = text_splitter.split_documents(docs)
        # for i, d in enumerate(docs):
        #     d.metadata['namespace'] = namespace
        return docs
    

    def read_doc(self, directory):
        file_loader = PyPDFDirectoryLoader(directory)
        document = file_loader.load()

        return document
    
    def add_vector_to_index(self, files_dir, index, namespace=None):
        print("...Loading Data")
        documents = self.chunk_data(self.read_doc(files_dir), namespace=namespace)
        print(len(documents))
        uuids = [str(uuid4()) for _ in range(len(documents))]
        index = self.get_index(index)
        vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
        vector_store.add_documents(documents=documents, ids=uuids) #  use the documents id instead
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
        
        # prompt = PromptTemplate(template=INSTRUCTIONS, input_variables=['agent_scratchpad', 'input'])
        prompt = hub.pull("hwchase17/openai-functions-agent")

        # corousels_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'corousels'}},)
        # podcast_retriever = self.vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'podcast'}},)

        # podcast_tool = create_retriever_tool(
        #     podcast_retriever,
        #     "podcast_summary",
        #     "If you find relevant information about Podcast use this tool to give a summary"
        # )


        # corousels_tool = create_retriever_tool(
        #     retriever=corousels_retriever,
        #     name="corousels_generator",
        #     description="Use this for social media or content generation related queries",
        #     document_prompt=COROUSEL_INSTRUCTIONS
        #     )
        
        # podcast_tool = create_retriever_tool(
        #     None,
        #     "podcast_summary",
        #     'If the questions is out of the scope: simply answer with "Out of scope! I cant give you an answer to that'
        # )

        # tools = [ corousels_tool]
        tools = self.build_tools()

        agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        self.agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools, max_iterations=25)

        # return self.agent_executor
    def build_tools(self):
        tools = []
        for index in self.indices:
            name = index['name']
            description = index['description']
            document_template = index['document_prompt']
            print(f'\n{document_template}')
            
            index = self.get_index(name)
            vector_store = PineconeVectorStore(index=index, embedding=self.embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 4, }, )#'filter': {'namespace': 'corousels'}},)
            p = PromptTemplate(
                input_variables=['context'],
                template=document_template
                )
            if 'context' not in p.input_variables:
                print("Nothing was found")
                print(p.input_variables)
            else:
                print("Was found")
            tool = create_retriever_tool(retriever=retriever, 
                                         name=name, 
                                         description=description, 
                                         document_prompt=p)
            tools.append(tool)
        return tools

    def query_response(self, query):

        result = self.agent_executor.invoke({'input': query})
        return result['output']



    

assistant_tool = AssistantTool()
# tools = assistant_tool.build_tools()
# print("Tools count: ",len( tools ))
# print(tools)
# assistant_tool.delete_index_content('transcripts')
# assistant_tool.add_vector_to_index('transcripts', 'transcripts')


print(assistant_tool.query_response("What did Jakob and Jenn Miller talk about"))
# print(assistant_tool.query_response("Need a content about Building sustainable funding strategy"))

# main(assistant_tool.vector_store)