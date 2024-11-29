# import langchain.text_splitter
# import openai
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
# from langchain.agents import Tool
load_dotenv()

transcripts_dir = 'transcripts'
documents_dir = 'documents'
corousels_dir = ''
book_drafts_dir = ''

api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_DEFAULT_API_KEY_ATHEM")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")


def clear_console():
    os.system('clear')

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    document = file_loader.load()

    return document

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)
    for i, d in enumerate(docs):
        d.metadata['from'] = 'jacobs podcast'
        d.metadata['title'] = d.page_content
    return docs


def add_vector_to_index(vector_store):
    print("...Loading Data")
    documents = chunk_data(read_doc(transcripts_dir))
    print(len(documents))
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    print("Done!")



def get_index(pc, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=1536, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    index = pc.Index(index_name)
    # print("Index was created successfully")
    return index 
    

def query_vector_store(vector_store, prompt):
    response = vector_store.similarity_search(prompt, k=2) # I can also include filters= {'type': podcast} or anything the metadata has
    # for res in response:
    #     print(f'\n{res.metadata}\n, {res.page_content}, ')
    return response


clear_console()
embeddings = OpenAIEmbeddings(api_key=api_key)
# vectors = embeddings.embed_query("And this is embedding two lets see if anything changes")

# Establish connection with pinecone
pc = Pinecone(
        api_key=pinecone_api_key
    )

index_name = 'transcripts'
index = get_index(pc, index_name)


vector_store = PineconeVectorStore(index=index, embedding=embeddings)
# add_vector_to_index(vector_store=vector_store)  # Only when there's new data






def main(vector_store):
    prompt = hub.pull("hwchase17/openai-functions-agent")
    print(prompt)
    llm = ChatOpenAI(
    api_key=api_key, 
    model='gpt-4', 
    temperature=0.7
    )
    # print(llm.invoke("What model are you?"))
    # vector_store.as_retriever()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    # print(retriever.invoke("""I'm Scandinavian and I like to have things very informal and conversational, which means I'll
    # make an intro. Is that Louis? that how I pronounce it? """))

    retriever_tool = create_retriever_tool(
        retriever,
        "podcast_summary",
        "______INSTRUCTION_________"
    )


    tools = [retriever_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)


    while True:
        question = input("Question: ")
        if question in ['q', 'quit']:
            break
        result = agent_executor.invoke({'input': question})
        clear_console()
        print(result['output'])

main(vector_store)