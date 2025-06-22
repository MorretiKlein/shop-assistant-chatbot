import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from qdrant_client import QdrantClient
from typing import Dict, Optional, List
import mysql.connector

client = QdrantClient(path="./langchain_qdrant") 

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = './vietnamese-bi-encoder')


class HybridProductRetriever:
    """hybrid search both retriever and SQL"""
    
    def __init__(self, client, embeddings, db_config, collection_name="base_knowledge"):
        self.client = client
        self.embeddings = embeddings
        self.db_config = db_config
        self.collection_name = collection_name
    
    def hybrid_search(self, query_text: str, 
                     top_k: int = 10,
                     filters: Optional[Dict] = None) -> List[Dict]:
        
        query_vector = self.embeddings.embed_query(query_text)
        
        search_params = {
            "collection_name": self.collection_name,
            "query": query_vector,
            "limit": top_k,
            "with_payload": True,
            "with_vectors": True
        }
        
        if filters:
            search_params["query_filter"] = self._build_qdrant_filter(filters)
        
        qdrant_results = self.client.query_points(**search_params)
        ids = [p.payload["id"] for p in qdrant_results.points]
        
        if not ids:
            return []
        
        # get from SQL
        full_products = self.get_full_product_details(ids)
        
        # sort relevance score
        product_dict = {p["id"]: p for p in full_products}
        ordered_results = []
        
        for hit in qdrant_results.points:
            product_id = hit.payload["id"]
            if product_id in product_dict:
                product = product_dict[product_id]
                product["relevance_score"] = hit.score
                ordered_results.append(product)
        
        return ordered_results
    
    def _build_qdrant_filter(self, filters: Dict):
        """Xây dựng Qdrant filter từ dict"""
        from qdrant_client.http import models
        
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Range filter: {"price": {"gte": 100, "lte": 1000}}
                if "gte" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            range=models.Range(gte=value["gte"])
                        )
                    )
                if "lte" in value:
                    conditions.append(
                        models.FieldCondition(
                            key=field,
                            range=models.Range(lte=value["lte"])
                        )
                    )
            else:
                # Exact match
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                )
        
        return models.Filter(must=conditions)
    
    def get_full_product_details(self, product_ids: List[int]) -> List[Dict]:
        """Lấy thông tin chi tiết sản phẩm từ SQL"""
        if not product_ids:
            return []
        
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            
            placeholders = ','.join(['%s'] * len(product_ids))
            
            query = f"""
        SELECT id, name, description, image_url, price, discount_percent
        FROM products 
        WHERE id IN ({placeholders}) and is_available_online = 1
            """
            
            cursor.execute(query, product_ids)
            products = cursor.fetchall()
            
            # Chuyển đổi Decimal và datetime thành string nếu sau này thêm date
            for product in products:
                for key, value in product.items():
                    if hasattr(value, 'isoformat'):  # datetime
                        product[key] = value.isoformat()
                    elif hasattr(value, '__float__'):  # Decimal
                        product[key] = float(value)
            
            conn.close()
            return products
            
        except Exception as e:
            print(f"Error fetching product details: {str(e)}")
            return []

