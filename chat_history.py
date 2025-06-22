import os
import json
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage


def message_to_dict(message):
    """
    Chuyển đổi một tin nhắn sang dạng dictionary.
    """
    if isinstance(message, HumanMessage):
        return {'type': 'human', 'content': message.content, 'additional_kwargs': message.additional_kwargs}
    elif isinstance(message, AIMessage):
        return {'type': 'ai', 'content': message.content, 'additional_kwargs': message.additional_kwargs}
    return {}


def dict_to_message(message_dict):
    """
    Chuyển đổi dictionary thành tin nhắn phù hợp.
    """
    if message_dict['type'] == 'human':
        return HumanMessage(content=message_dict['content'], additional_kwargs=message_dict.get('additional_kwargs', {}))
    elif message_dict['type'] == 'ai':
        return AIMessage(content=message_dict['content'], additional_kwargs=message_dict.get('additional_kwargs', {}))
    return None


def update_conversation(store, category = "product_question", file_path="conversation_data.json"):
    """
    Cập nhật lịch sử hội thoại vào file JSON theo danh mục.
    """
    conversation = {}

    folder_chat_history = "chat_history/"
    os.makedirs(folder_chat_history, exist_ok=True)
    file_path = os.path.join(folder_chat_history, file_path)


    # Chuyển đổi `store` thành dạng dictionary để lưu
    for user_id, chat_history in store.items():
        conversation[user_id] = {
            category: [message_to_dict(msg) for msg in chat_history.messages]
        }

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)

        # Hợp nhất dữ liệu mới với dữ liệu cũ
        for user_id, new_data in conversation.items():
            if user_id not in existing_data:
                existing_data[user_id] = new_data
            else:
                if category not in existing_data[user_id]:
                    existing_data[user_id][category] = new_data[category]
                else:
                    # Thêm tin nhắn mới mà không trùng lặp
                    for message in new_data[category]:
                        if message not in existing_data[user_id][category]:
                            existing_data[user_id][category].append(message)
    else:
        existing_data = conversation

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)


def load_previous_conversation(user_id, category = "product_question", file_path="conversation_data.json"):
    """
    Tải lịch sử hội thoại từ file JSON cho một user và danh mục cụ thể.
    """

    folder_chat_history = "chat_history/"
    os.makedirs(folder_chat_history, exist_ok=True)
    file_path = os.path.join(folder_chat_history, file_path)

    if not os.path.exists(file_path):
        print("No conversation data file found.")
        return None, None

    with open(file_path, 'r', encoding='utf-8') as file:
        conversation_data = json.load(file)

    if user_id not in conversation_data or category not in conversation_data[user_id]:
        print(f"No chat history found for user ID: {user_id} and category: {category}")
        return None, None

    user_messages = conversation_data[user_id][category]
    if not user_messages:
        return None, None

    first_message = dict_to_message(user_messages[0])
    recent_messages = [dict_to_message(msg) for msg in user_messages[-5:]]
    return first_message, recent_messages


def initialize_session_from_history(chat_history, first_message, recent_messages):
    """
    Khởi tạo lịch sử hội thoại từ dữ liệu đã tải.
    """
    if first_message and not any(msg.content == first_message.content for msg in chat_history.messages):
        chat_history.add_message(first_message)

    for msg in recent_messages:
        if msg and not any(existing_msg.content == msg.content for existing_msg in chat_history.messages):
            chat_history.add_message(msg)
