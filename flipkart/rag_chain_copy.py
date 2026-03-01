from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from flipkart.config import Config

class RAGChainBuilder:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.model = ChatGroq(model=Config.RAG_MODEL, temperature=0.5)
        self.history_store = {}

    def _get_history(self, session_id: str):
        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()
        return self.history_store[session_id]
    
    def build_chain(self):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an e-commerce assistant. Use the context to answer.\n\nCONTEXT:\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"), 
            ("human", "{input}")  
        ])

        # SUCCESSFUL PATTERN: 
        # 1. Take the input dict, 2. Add 'context' to it, 3. Send to prompt
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x["input"])
            )

            | qa_prompt 
            | self.model 
            | StrOutputParser()
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
