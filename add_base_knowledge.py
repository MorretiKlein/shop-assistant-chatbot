from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings
import mysql.connector
import os
from typing import Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ProductIndexer:
    def __init__(self):
        self.client = QdrantClient(path="./langchain_qdrant")
        self.embeddings = HuggingFaceEmbeddings(model_name='./vietnamese-bi-encoder')
        self.collection_name = "base_knowledge"
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123',
            'database': 'myshop'
        }

    def create_text_description(self, product_data: Dict) -> str:
        fields = [
            product_data.get('name', ''),
            product_data.get('description', ''),
            product_data.get('image_url', ''),
            f"giá {product_data.get('price', 0)}",
            f"chiết khấu {product_data.get('discount_percent', 0)}"
        ]
        return ". ".join([f for f in fields if f])

    def index_products(self):
        # Tạo collection nếu chưa có
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )

        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, name, description, image_url, price, discount_percent
            FROM products 
            WHERE is_available_online = 1
        """)
        products = cursor.fetchall()

        points = []
        for i, row in enumerate(products):
            product_data = {
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'image_url': row[3],
                'price': row[4],
                'discount_percent': row[5],
            }

            text = self.create_text_description(product_data)
            vector = self.embeddings.embed_query(text)

            payload = {
                "id": product_data["id"],
                "name": product_data["name"],
                "price": float(product_data["price"]),
                "discount_percent": float(product_data["discount_percent"])
            }

            points.append(models.PointStruct(id=i, vector=vector, payload=payload))

        self.client.upsert(collection_name=self.collection_name, wait=True, points=points)
        conn.close()
        print(f"Indexed {len(points)} products to Qdrant.")

if __name__ == "__main__":
    indexer = ProductIndexer()
    indexer.index_products()
