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

        # system = (
        #     "You are an e-commerce assistant. Use the provided context to answer the user's query. "
        #     "Only If you are asked about product, then Display the output in this strict Markdown format:\n\n"
        #     "1. Main Title: Use '# **Recommended**' followed by '---'.\n"
        #     "2. Item Headings: Use '### Name'.\n"
        #     "3. Details: Use bullet points for **Product Name**, **Description**, **Price**, and **Availability**.\n"
        #     "4. Links: If a URL is in the context, format it as [Link Text](URL). Do not ignore them.\n"
        #     "5. Constraints: No ancillary text or introductory remarks. Only output the structured markdown.\n\n"
        #     "CONTEXT:\n{context}"
        # )

        system = (
                    "You are an e-commerce assistant. Use the provided context to answer the user's query."
                    "Do not use code blocks (like ```markdown) to wrap your answer. Output the raw markdown syntax directly.\n\n"
                    "STRATEGY:\n"
                    "1. If the user asks for product recommendations or details, use the STRICT MARKDOWN format below.\n"
                    "2. If the user asks a general question or the context is irrelevant, answer naturally without the product table.\n"
                    "3. If the context does not contain the answer, say you don't know—do not make up product details.\n\n"
                    "STRICT MARKDOWN FORMAT:\n"
                    "### **Recommended**\n---\n"
                    "##### Name\n"
                    "* **Product Name**: [Name]\n"
                    "* **Description**: [Description]\n"
                    "* **Price**: [Price]\n"
                    "* **Availability**: [Availability]\n\n"
                    "CONTEXT:\n{context}"
                )

        qa_prompt = ChatPromptTemplate.from_messages([
            # ("system", "You are an e-commerce assistant. Use the context to answer and display the output as markdown.\n\nCONTEXT:\n{context}"),
            ("system", system),
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
