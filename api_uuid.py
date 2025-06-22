import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mysql.connector
import uvicorn

from data_filter import *
from rag_chatbot import  answer_user
app = FastAPI()

from langchain.globals import set_verbose
set_verbose(False)


db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '123',
    'database': 'myshop'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

class UserRegister(BaseModel):
    username: str

class UserLogin(BaseModel):
    username: str

class askData(BaseModel):
    username: str
    question: str

class TextData(BaseModel):
    title: str
    text: str
    username: str


def create_user_table_if_not_exists(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) UNIQUE NOT NULL
        )
    """)

# 1. API tạo username, password và uuid, và thêm vào database
@app.post("/register")
def register(user: UserRegister):
    db = get_db_connection()
    cursor = db.cursor()
    try:
        create_user_table_if_not_exists(cursor)
        cursor.execute("INSERT INTO user (username) VALUES (%s)",(user.username,))
        db.commit()
        return {"message": "User registered successfully", "uuid": user.username}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        cursor.close()
        db.close()
        

@app.delete("/delete_user")
def delete_user(user: UserRegister):
    result = get_user_from_db(user.username)
    if not result:
        raise HTTPException(status_code=404, detail="User not found")
    db = get_db_connection()
    cursor = db.cursor()
    try:
        cursor.execute("DELETE FROM user WHERE username = %s", (user.username,))
        db.commit()
        return {"message": "User deleted successfully"}
    except mysql.connector.Error as err:
        raise HTTPException(status_code=500, detail=str(err))
    finally:
        cursor.close()
        db.close()


def get_user_from_db(username: str):
    """Truy xuất thông tin người dùng từ cơ sở dữ liệu."""
    db = get_db_connection()
    cursor = db.cursor()
    try:
        create_user_table_if_not_exists(cursor)
        cursor.execute("SELECT username FROM user WHERE username = %s", (username,))
        return cursor.fetchone() 
    finally:
        cursor.close()
        db.close()

@app.post("/ask_user")
def ask_question(data: askData):
    result = get_user_from_db(data.username)
    if not result :  
        return {"message": "User not found"}
    
    username = result[0]
    question = data.question
    answer = answer_user(question, username)
    return {"message": answer}


# class addQA(BaseModel):
#     subject: str
#     question: str
#     answer: str


# @app.post("/add_base_QA")
# def add_qa(data: addQA):
#     try:
#         text = [f"{data.question}\n{data.answer}"]
#         qa_dict = {
#             "subject": data.subject,
#             "question": data.question,
#             "answer": data.answer
#         }
#         file_path = "qa_data.txt"
#         with open(file_path, "a", encoding="utf-8") as f:
#             f.write(str(qa_dict) + "\n")
#         return {"message": "add QA successfully"}
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
#kill $(lsof -t -i:8000)
