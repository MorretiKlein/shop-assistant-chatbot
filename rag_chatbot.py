from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_qdrant import Qdrant
from langchain_community.vectorstores import Qdrant
from data_filter import client
from chat_history import *
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import uuid
from qdrant_client.http.models import Distance, VectorParams
import mysql.connector

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Optional, List
from data_filter import *
from langchain_core.runnables import RunnableLambda

from langchain.globals import set_verbose
set_verbose(False)

class RangeFilter(BaseModel):
    gte: Optional[float] = Field(..., description="Minimum value of the field")
    lte: Optional[float] = Field(..., description="Maximum value of the field")

class FilterSchema(BaseModel):
    """Represents a filter with a range for a specific product field"""
    price: Optional[RangeFilter] = None
    discount_percent: Optional[RangeFilter] = None

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key = "")
structured_llm = llm.with_structured_output(FilterSchema)

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name='./vietnamese-bi-encoder')
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',
    'database': 'myshop'
}


def extract_filter_from_message(message: str) -> dict:
    """Trích xuất filter từ tin nhắn của user"""
    result = structured_llm.invoke(message)
    return result.dict(exclude_none=True)

hybrid_retriever = HybridProductRetriever(client, embeddings, DB_CONFIG)

def get_vector_store(collection_name):
    """Tạo và trả về vector store cho collection (cho general knowledge)"""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"content": VectorParams(size=768, distance=Distance.COSINE)}
        )
    vectorstore = Qdrant(
        client=client, 
        collection_name=collection_name, 
        embeddings=embeddings, 
        vector_name="content"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def format_product_info(products: List[Dict]) -> str:
    """Format thông tin sản phẩm thành text cho context"""
    if not products:
        return "Không tìm thấy sản phẩm phù hợp."
    
    formatted_products = []
    for product in products:
        product_text = f"""
        Sản phẩm: {product.get('name', 'N/A')}
        Giá: {product.get('price', 'N/A'):,} VND
        Giảm giá: {product.get('discount_percent', 0)}%
        Mô tả: {product.get('description', 'N/A')}
        Tồn kho: {product.get('stock_quantity', 'N/A')}
        Độ liên quan: {product.get('relevance_score', 0):.3f}
        """.strip()
        formatted_products.append(product_text)
    
    return "\n\n".join(formatted_products)

def prompt_subject():
    """Prompt chính cho tư vấn viên"""
    return """Bạn là một tư vấn viên chuyên nghiệp, có trách nhiệm hỗ trợ khách hàng bằng cách cung cấp câu trả lời chi tiết, thân thiện và thuyết phục cho các câu hỏi về sản phẩm. Mục tiêu của bạn là hiểu rõ nhu cầu của khách hàng và đề xuất sản phẩm hoặc giải pháp phù hợp.

    Hãy sử dụng thông tin sản phẩm chi tiết được cung cấp để đưa ra lời tư vấn chính xác. Khi giới thiệu sản phẩm, hãy đề cập đến:
    - Tên sản phẩm và ID
    - Giá cả và ưu đãi (nếu có)
    - Các tính năng nổi bật
    - Đánh giá từ khách hàng khác
    - Tình trạng tồn kho

    Sử dụng ngôn ngữ đơn giản, dễ hiểu để xây dựng lòng tin và khuyến khích khách hàng đưa ra quyết định mua hàng.

    Thông tin kiến thức chung: {general_context}

    Thông tin sản phẩm chi tiết từ hệ thống:
    {product_context}
    """

def answer_user(formed_question, user_id):
    """Trả lời câu hỏi của user với hybrid search"""
    # Lưu trữ lịch sử chat
    store = {}

    
    first_message, recent_messages = load_previous_conversation(user_id, category = "product_question", file_path =f"{user_id}.txt")

    if user_id not in store:
        store[user_id] = InMemoryChatMessageHistory()
    if first_message is not None and recent_messages is not None:
        initialize_session_from_history(store[user_id], first_message, recent_messages)
    
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]
    print("store:", store[user_id])

    # Prompt cho việc contextualize câu hỏi
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # Custom function dual retrieval
    def dual_retrieval_answer(inputs):
        """Thực hiện retrieval từ cả general knowledge và product database"""
        
        question = inputs["input"]
        chat_history = inputs.get("chat_history", [])

        print(f"Chat history length: {len(chat_history)}")
        print(f"Last message: {chat_history[-1] if chat_history else 'No history'}")
    
        
        # 1. Contextualize question nếu có lịch sử
        contextual_question = question
        if chat_history:
            try:
                contextualized = llm.invoke(
                    contextualize_q_prompt.format_messages(
                        chat_history=chat_history,
                        input=question
                    )
                )
                contextual_question = contextualized.content
            except:
                contextual_question = question
        
        try:
            filters = extract_filter_from_message(contextual_question)
        except:
            filters = {}
        
        relevant_products = hybrid_retriever.hybrid_search(
            query_text=contextual_question,
            top_k=5,
            filters=filters if filters else None
        )
        
        general_retriever = get_vector_store("base_knowledge")  # Collection cho kiến thức chung
        try:
            general_docs = general_retriever.get_relevant_documents(contextual_question)
            general_context = "\n".join([doc.page_content for doc in general_docs])
        except:
            general_context = "Không có thông tin chung phù hợp."
        
        product_context = format_product_info(relevant_products)
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_subject()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        formatted_messages = qa_prompt.format_messages(
            general_context=general_context,
            product_context=product_context,
            chat_history=chat_history,
            input=question
        )
        
        response = llm.invoke(formatted_messages)
        
        return {
            "answer": response.content,
            "source_products": relevant_products,
            "general_context": general_context
        }

    # conversational chain với message history
    conversational_chain = RunnableWithMessageHistory(
        RunnableLambda(dual_retrieval_answer),  # ✅ bọc lại ở đây
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Gọi chain và lấy kết quả
    try:
        result = conversational_chain.invoke(
            {"input": formed_question},
            config={
                "configurable": {"session_id": user_id}
            },
        )
        
        answer = result["answer"]
        update_conversation(store, category = "product_question", file_path =f"{user_id}.txt")
        return answer.strip()
        
    except Exception as e:
        print(f"Error in answer_user: {str(e)}")
        return "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này. Vui lòng thử lại sau."

def test_hybrid_search(query: str):
    """Test hybrid search function"""
    results = hybrid_retriever.hybrid_search(query, top_k=3)
    print(f"Query: {query}")
    print(f"Found {len(results)} products:")
    for product in results:
        print(f"- {product['name']} (ID: {product['id']}) - Score: {product.get('relevance_score', 0):.3f}")
    return results