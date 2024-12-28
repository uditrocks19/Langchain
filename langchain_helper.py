import config
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import  OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
api_key = config.api_key
embeddings = OpenAIEmbeddings(openai_api_key = api_key)

# function LLM
video_url = "https://youtu.be/BispW3eFsPg?si=2n6pTS0UOfuVdyFk"
def create_vector_db_from_youtube_url(video_url : str) ->FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000 , chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs , embeddings)
    return db

# get response from query
# k similarity check
def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")  # Updated model
    prompt = PromptTemplate(
        input_variables=['question', 'docs'],
        template="""You are a YouTube assistant
        that helps users by answering questions based on a video transcript.
        Answer the following question: {question}
        by searching the video transcript: {docs}.
        If you don't have enough information, say "I don't know, dear."
        Your answer should be 4 to 5 lines."""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs=docs_page_content)
    return response.replace('\n', '')

# Example usage
# db = create_vector_db_from_youtube_url(video_url)
# print(get_response_from_query(db, query="Summarise the podcast?", k=4))