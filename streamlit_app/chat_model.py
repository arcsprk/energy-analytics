from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.schema.document import Document
from typing import List

llm = ChatOllama(base_url="http://localhost:11434", model="phi3")
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=80,
    memory_key="chat_history",
    return_messages=True,
)

def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["chat_history"]

    
class CustomRetriever(BaseRetriever):
    """Always return three static documents for testing."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [
            Document(page_content="Japan has a population of 126 million people.", metadata={"source": "https://en.wikipedia.org/wiki/Japan"}),
            Document(page_content="Japanese people are very polite.", metadata={"source": "https://en.wikipedia.org/wiki/Japanese_people"}),
            Document(page_content="United States has a population of 328 million people.", metadata={"source": "https://en.wikipedia.org/wiki/United_States"}),
            ]

        # return [Document(page_content=query)]

class ChatWithCustomRetriver():
    def __init__(self):
        # self.model = llm
        self.model = ChatOllama(base_url="http://localhost:11434", model="mistral", streaming=True)


        prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If the question is to request links, please only return the source links with no answer.
        2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
        3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

        {context}

        Question: {question}
        Helpful Answer:"""

        self.prompt = PromptTemplate.from_template(prompt_template) # prompt_template defined above
        self.retriever = CustomRetriever()

        # self.chain = ({"question": RunnablePassthrough()}
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser())

        # llm = ChatOpenAI()
        # llm = ChatOllama(base_url="http://localhost:11434", model="phi3", streaming=True)

        # self.llm_chain = LLMChain(llm=self.model, prompt=self.prompt, callbacks=None, verbose=True)
        # document_prompt = PromptTemplate(
        #     input_variables=["page_content", "source"],
        #     template="Context:\ncontent:{page_content}\nsource:{source}",
        # )
        # combine_documents_chain = StuffDocumentsChain(
        #     llm_chain=self.llm_chain,
        #     document_variable_name="context",
        #     document_prompt=document_prompt,
        #     callbacks=None,
        # )

        
        # retriever = CustomRetriever()

        # self.chain_qa = RetrievalQA(
        #     combine_documents_chain=combine_documents_chain,
        #     callbacks=None,
        #     verbose=True,
        #     retriever=retriever,
        #     return_source_documents=True,
        # )


    def ask(self, query: str):
        
        # result = self.chain_qa.invoke(query)

        result = self.chain.invoke(query)

        # memory.save_context(
        #     {"input": query},
        #     {"output": result.content},
        # )

        return result



class ChatTable:
    def __init__(self):
        # self.model = llm
        # self.model = ChatOllama(base_url="http://localhost:11434", model="phi3", streaming=True)
        self.model = ChatOllama(base_url="http://localhost:11434", model="mistral", streaming=True)

        # self.prompt = PromptTemplate.from_template(
        #     """
        #     Instruction: You are an assistant for table question-answering tasks. Use the following pieces of retrieved context 
        #     to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
        #      maximum and keep the answer concise. 

        #     Context: {context} 
        #     Question: {question}             
        #     Answer: 
        #     """
        # )

        self.prompt = PromptTemplate.from_template(
            """
            Instruction: You are an assistant for table question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. 

            Question: {question}             
            Answer: 
            """
        )

        self.chain = ({"question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | StrOutputParser())


    def ask(self, query: str):
        
        result = self.chain.invoke(query)
        # memory.save_context(
        #     {"input": query},
        #     {"output": result.content},
        # )

        return result





class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):


        self.model = llm

        self.embedding_model = FastEmbedEmbeddings()

        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest_pdf(self, file_path: str):
        docs = PyPDFLoader(file_path=file_path).load()
 
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embedding_model)

        # Define the path for generated embeddings
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        vector_store.save_local(DB_FAISS_PATH)

        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
        
         # Create a conversational chain
        # self.chain = ConversationalRetrievalChain.from_llm(llm=self.model, retriever=self.retriever)


    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        result = self.chain.invoke(query)
        memory.save_context(
            {"input": query},
            {"output": result.content},
        )

        return result

        # return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None


class ChatCSV:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):

        # llm = Ollama(base_url="http://localhost:11434", model="mistral")
        # self.model = ChatOllama(model="mistral")
        # self.model = ChatOllama(base_url="http://localhost:11434", model="phi3")
        self.model = llm
        # self.embedding_model = OllamaEmbeddings(base_url="http://localhost:11434", model="mistral")
        self.embedding_model = FastEmbedEmbeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved table 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest_csv(self, file_path: str):
        docs = CSVLoader(file_path=file_path).load()
 
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = FAISS.from_documents(documents=chunks, embedding=self.embedding_model)
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())


    def ask(self, query: str):
        if not self.chain:
            return "Please, add a CSV file first."
        result = self.chain.invoke(query)
        memory.save_context(
            {"input": query},
            {"output": result.content},
        )

        return result

        # return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None