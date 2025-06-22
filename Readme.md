
# Shop assistant by Trungtran

### Install environment
install anacoda
``` shell
conda create -n uuid python=3.11 -y
conda activate uuid

```
```shell
pip install -r requirements.txt
```

You can edit qdrant-client to qdrant-client[fastembed-gpu] or vice versa, depending on the machine, I recommend using qdrant-client when it is effective enough

```shell
run : python api_uuid.py
kill port: kill $(lsof -t -i:8000)
```
(uvicorn main:app --reload --port 9000)

2. **Resorce document**
have example.txt as a resource document
# API Usage Guide

## Endpoints

### 1. Register
- **Route**: `/register`
- **Parameters**:
  - `username`: `string` - User's name
  - `password`: `string` - Password

### 2. Ask
- **Route**: `/ask_user`
- **Parameters**:
  - `username`: `string` - User's name
  - `question`: `string` - Question you want to send to the AI
- **Parameters**:
  - `return {"message": answer}`

### 3. Delete user
- **Route**: `/delete_user`
- **Description**: Deletes user.

ngrok http http://localhost:8000

