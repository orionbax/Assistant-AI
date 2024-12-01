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
from pprint import pprint

# from langchain.agents import Tool
load_dotenv()

transcripts_dir = 'transcripts'
documents_dir = 'documents'
corousels_dir = 'corousels_txt'
book_drafts_dir = ''

api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_DEFAULT_API_KEY_ATHEM")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")

PODCAST_INSTRUCTION="Search for information regarding the given prompt or word about Jakobs or his Podcast, for any information about Jakobs or his Podcast, You must use this tool!"
LINKED_IN="For any reports that has to do with generating content for linked in use this tool, You must use this tool for anything relate to generating new contents for linked in!"


def clear_console():
    os.system('clear')

def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    document = file_loader.load()

    return document

def chunk_data(docs, chunk_size=800, chunk_overlap=50, namespace=""):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(docs)
    for i, d in enumerate(docs):
        d.metadata['namespace'] = namespace
    return docs


def add_vector_to_index(vector_store, files_dir, namespace):
    print("...Loading Data")
    documents = chunk_data(read_doc(files_dir), namespace=namespace)
    print(len(documents))
    # pprint(documents)
    # uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents) #, ids=uuids)  use the documents id instead
    print("Done!")



def get_index(pc, index_name):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=3072, 
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



def main(vector_store):
    prompt = hub.pull("hwchase17/openai-functions-agent")
    # prompt = PromptTemplate.from_template(
    #     """{agent_scratchpad}
    #     You should always tell jokes at the end of your sentences.
    #     """
    # )
    # print(prompt)
    prompt_template = """
    You are a helpful assistant that doesn't go out of the given scope.
    You are here to assist with generating new content from the user.
    Any question out of your scope should be answered with -> Your question is out of my scope.

    Context so far:
    {agent_scratchpad}

    User query:
    {input}

    Respond accordingly based on the above context.
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['agent_scratchpad', 'input'])

    # print(prompt)
    # pprint(dir(prompt))
    # pprint(prompt.to_json())
    llm = ChatOpenAI(
    api_key=api_key, 
    model='gpt-4', 
    temperature=0.7,
    max_tokens=1000,
    )
    podcast_retriever = vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'podcast'}},)
    
    corousels_retriever = vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'corousels'}},)

    podcast_tool = create_retriever_tool(
        podcast_retriever,
        "podcast_summary",
        "If you find relevant information about Podcast use this tool to give a summary"
    )

    # corousels_tool = create_retriever_tool(
    #     corousels_retriever,
    #     "corousels_generator",
    #     "Use this tool to answer anyquestions asked about corousels, When you find relevant informations in the tool you must let the user know about your reference."
    # )

    corousels_tool = create_retriever_tool(
        corousels_retriever,
        "corousels_generator",
        """ Use this tool if asked to generate new linked content or corousel, 
        Your answer needs to be as clear and as detailed as possible,
        Make sure you state the basis of your response, such as resources you have found.
        finally your response should be delivered in a json format as such and no content should be out of its place
            {
            'content_for': linked_in
            "Title": "Put the tile here"
            "content": "CONTENT GOES IN HERE, NEEDS TO BE AS DETAILED AS POSSIBLE",
            "caption": "Put the caption here",
            "hashtags": "#Hashtag1 #hashtag2 etc",
            "Summary": If you think is necessary,
            "resources" The basis of your answer, could be file name or could be anything just make it clear wether it was from your memory or outside source! 
            }
        """
    )

    podcast_tool = create_retriever_tool(
        None,
        "podcast_summary",
        'If the questions is out of the scope: simply answer with "Out of scope! I cant give you an answer to that'
    )

    tools = [podcast_tool, corousels_tool]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)


    while True:
        question = input("Question: ")
        if question in ['q', 'quit']:
            break
        result = agent_executor.invoke({'input': question})
        # result = corousels_retriever.invoke(question)
        # pprint(result)
        # clear_console()
        print(result['output'])

def test_agent(vector_store):
    from langchain.chains.llm  import LLMChain, Runnable
    # from langchain.chains.retrieval_qa import RetrievalQAWithSourcesChain
    from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
    # from langchain.chat_models import ChatOpenAI

    # Define the retrieval-based QA chain
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Custom prompt for LinkedIn content generation
    prompt_template = """
    You are an AI assistant. Given the following context from past LinkedIn posts:

    {context}

    Create a new, engaging LinkedIn post on the topic: "{topic}". Ensure the tone is professional and the post aligns with the style of the provided context.
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "topic"])

    # Set up the QA chain
    llm = ChatOpenAI(model="gpt-4", api_key=api_key)
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff",
    #     return_source_documents=True
    # )
    llm_chain = LLMChain(llm=llm, prompt=prompt)

#     # Use the LLM chain in the RetrievalQA chain
    chain = RetrievalQAWithSourcesChain(
    combine_documents_chain=llm_chain,
    retriever=retriever,
    return_source_documents=True,
)
#     query = f"Create a post about Real engagement"
#     result = chain.run(query)
#     print(result)
#     return result


clear_console()
# embeddings = OpenAIEmbeddings(api_key=api_key)
embeddings = OpenAIEmbeddings(api_key=api_key,model='text-embedding-3-large')

# vectors = embeddings.embed_query("And this is embedding two lets see if anything changes")

# Establish connection with pinecone
pc = Pinecone(
        api_key=pinecone_api_key
    )

index_name = 'corousels'
index = get_index(pc, index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)
# podcast_retriever = vector_store.as_retriever(search_kwargs={"k": 4, 'filter': {'namespace': 'corousels'}},)
# print(podcast_retriever.invoke("Real engagement"))
# vector_store.add_documents()



# add_vector_to_index(vector_store, corousels_dir, namespace=index_name)  # Only when there's new data

# retriever = vector_store.as_retriever()


# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


main(vector_store)





#git push --set-upstream origin main

# git branch -D {branch name} -> forcefully delete a branch
