#!/usr/bin/env python3
"""
ObjectBox Vector Database Query Script

Query and retrieve vectors from ObjectBox database with similarity search,
text search, and document retrieval capabilities.

Usage:
    python objectbox_query.py --database-dir ./mobile_database --query "your search text"
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


from objectbox import Entity, Id, Store, String, Float32Vector
from objectbox.model import *

from database.vector_entity import (
    VectorEntity64,
    VectorEntity128,
    VectorEntity256,
    VectorEntity384,
    VectorEntity512,
    VectorEntity768,
    VectorEntity1024,
    VectorEntity1536
)

try:
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
except ImportError:
    logger.error("LangChain not installed. Install with: pip install langchain langchain-community")
    exit(1)

class VectorSearchResult:
    """Container for search results with similarity scores."""
    def __init__(self, entity, score: float):
        self.id = entity.id
        self.name = entity.name
        self.content = entity.content
        self.document = entity.document
        self.embedding = entity.embedding
        self.metadata = json.loads(entity.metadata) if entity.metadata else {}
        self.timestamp = entity.timestamp
        self.score = score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'content': self.content,
            'document': self.document,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'similarity_score': self.score
        }
    
    def __str__(self) -> str:
        """String representation for pretty printing."""
        return f"Score: {self.score:.4f} | Document: {self.document} | Content: {self.content[:100]}..."

class ObjectBoxQueryEngine:
    def __init__(self, database_dir: str, model_name: Optional[str] = None):
        self.database_dir = Path(database_dir)
        
        if not self.database_dir.exists():
            raise FileNotFoundError(f"Database directory not found: {database_dir}")
        
        # Load database info to get configuration
        info_file = self.database_dir / "database_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                self.db_info = json.load(f)
            self.embedding_dim = self.db_info['embedding_dimension']
            self.model_name = model_name or self.db_info['model_name']
            logger.info(f"Loaded database info: {self.embedding_dim}D embeddings, model: {self.model_name}")
        else:
            raise FileNotFoundError(f"Database info file not found: {info_file}")
        
        # Initialize embeddings (for query encoding)
        if model_name or self.model_name:
            logger.info(f"Initializing embeddings with model: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            self.embeddings = None
            logger.warning("No embedding model specified. Vector similarity search will not be available.")
        
        # Setup ObjectBox
        self.entity_class = self._get_entity_class(self.embedding_dim)
        self.model = Model()
        self._setup_model()
        
        # Open store in read-only mode
        self.store = Store(model=self.model, directory=str(self.database_dir), model_json_file=f"{self.database_dir}/objectbox-model.json")
        self.box = self.store.box(self.entity_class)
        
        logger.info(f"ObjectBox query engine initialized")
        logger.info(f"Database: {self.database_dir}")
        logger.info(f"Entity class: {self.entity_class}")
        logger.info(f"Total vectors: {self.box.count()}")
    
    def _get_entity_class(self, embedding_dim: int):
        """Get the appropriate entity class based on embedding dimension."""
        entity_map = {
            64: VectorEntity64,
            128: VectorEntity128,
            256: VectorEntity256,
            384: VectorEntity384,
            512: VectorEntity512,
            768: VectorEntity768,
            1024: VectorEntity1024,
            1536: VectorEntity1536
        }
        return entity_map[embedding_dim]
    
    def _setup_model(self):
        """Setup the ObjectBox model."""
        entity_configs = {
            64: (VectorEntity64, IdUid(7, 1007), IdUid(1, 1)),
            128: (VectorEntity128, IdUid(7, 2007), IdUid(2, 2)),
            256: (VectorEntity256, IdUid(7, 3007), IdUid(3, 3)),
            384: (VectorEntity384, IdUid(7, 4007), IdUid(4, 4)),
            512: (VectorEntity512, IdUid(7, 5007), IdUid(5, 5)),
            768: (VectorEntity768, IdUid(7, 6007), IdUid(6, 6)),
            1024: (VectorEntity1024, IdUid(7, 7007), IdUid(7, 7)),
            1536: (VectorEntity1536, IdUid(7, 8007), IdUid(8, 8))
        }
        
        entity_class, last_prop_id, entity_id = entity_configs[self.embedding_dim]
        self.model.entity(entity_class)
        self.model.last_entity_id = entity_id
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Calculate dot product
        dot_product = np.dot(vec1_np, vec2_np)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vec1_np)
        magnitude2 = np.linalg.norm(vec2_np)
        
        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def vector_similarity_search(self, query_text: str, top_k: int = 10, 
                                min_score: float = 0.0) -> List[VectorSearchResult]:
        """
        Perform vector similarity search using query text.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of VectorSearchResult objects sorted by similarity score
        """
        if not self.embeddings:
            raise ValueError("No embedding model available for vector search")
        
        # Generate query embedding
        logger.info(f"Generating embedding for query: '{query_text}'")
        query_embedding = self.embeddings.embed_query(query_text)
        logger.info(f"Query embedding dimension: {len(query_embedding)}")
        
        # Get all vectors from database
        logger.info("Retrieving all vectors from database...")
        all_entities = self.box.get_all()
        logger.info(f"Found {len(all_entities)} entities in database")
        
        # Log first few entities for debugging
        for i, entity in enumerate(all_entities[:3]):
            logger.info(f"Entity {i+1}:")
            logger.info(f"  ID: {entity.id}")
            logger.info(f"  Name: {entity.name}")
            logger.info(f"  Document: {entity.document}")
            logger.info(f"  Content length: {len(entity.content)}")
            try:
                embedding_len = len(entity.embedding) if entity.embedding is not None else 'None'
            except:
                embedding_len = 'Error'
            logger.info(f"  Embedding length: {embedding_len}")
            logger.info(f"  Content preview: {entity.content[:100]}...")
        
        if not all_entities:
            logger.warning("No entities found in database!")
            return []
        
        # Calculate similarities
        results = []
        successful_comparisons = 0
        failed_comparisons = 0
        
        for entity in all_entities:
            try:
                if entity.embedding is None:
                    logger.warning(f"Entity {entity.id} has no embedding")
                    failed_comparisons += 1
                    continue
                
                if len(entity.embedding) != len(query_embedding):
                    logger.warning(f"Entity {entity.id} embedding dimension mismatch: {len(entity.embedding)} vs {len(query_embedding)}")
                    failed_comparisons += 1
                    continue
                
                similarity = self.cosine_similarity(query_embedding, entity.embedding)
                logger.debug(f"Entity {entity.id} similarity: {similarity:.4f}")
                
                if similarity >= min_score:
                    results.append(VectorSearchResult(entity, similarity))
                
                successful_comparisons += 1
                
            except Exception as e:
                logger.warning(f"Error calculating similarity for entity {entity.id}: {e}")
                failed_comparisons += 1
                continue
        
        logger.info(f"Similarity calculations: {successful_comparisons} successful, {failed_comparisons} failed")
        logger.info(f"Results above threshold ({min_score}): {len(results)}")
        
        # Sort by similarity score (descending) and return top k
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Log top results
        for i, result in enumerate(results[:5]):
            logger.info(f"Top result {i+1}: Score {result.score:.4f}, Document: {result.document}")
        
        return results[:top_k]
    
    def text_search(self, search_term: str, search_field: str = "content", 
                   case_sensitive: bool = False) -> List[VectorSearchResult]:
        """
        Perform text-based search in document content or names.
        
        Args:
            search_term: Text to search for
            search_field: Field to search in ('content', 'name', 'document')
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of VectorSearchResult objects (score = 1.0 for matches)
        """
        logger.info(f"Performing text search for '{search_term}' in field '{search_field}'")
        
        all_entities = self.box.get_all()
        results = []
        
        search_lower = search_term.lower() if not case_sensitive else search_term
        
        for entity in all_entities:
            try:
                if search_field == "content":
                    field_value = entity.content
                elif search_field == "name":
                    field_value = entity.name
                elif search_field == "document":
                    field_value = entity.document
                else:
                    logger.warning(f"Unknown search field: {search_field}")
                    continue
                
                if not case_sensitive:
                    field_value = field_value.lower()
                
                if search_lower in field_value:
                    results.append(VectorSearchResult(entity, 1.0))  # Perfect match score
                    
            except Exception as e:
                logger.warning(f"Error searching entity {entity.id}: {e}")
                continue
        
        return results
    
    def get_by_document(self, document_name: str) -> List[VectorSearchResult]:
        """
        Get all chunks from a specific document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            List of VectorSearchResult objects from the document
        """
        logger.info(f"Retrieving all chunks from document: '{document_name}'")
        
        all_entities = self.box.get_all()
        results = []
        
        for entity in all_entities:
            if entity.document == document_name:
                results.append(VectorSearchResult(entity, 1.0))
        
        # Sort by chunk index if available in metadata
        def get_chunk_index(result):
            try:
                return result.metadata.get('chunk_index', 0)
            except:
                return 0
        
        results.sort(key=get_chunk_index)
        return results
    
    def get_by_id(self, vector_id: int) -> Optional[VectorSearchResult]:
        """
        Get a specific vector by its ID.
        
        Args:
            vector_id: ID of the vector entity
            
        Returns:
            VectorSearchResult object or None if not found
        """
        try:
            entity = self.box.get(vector_id)
            if entity:
                return VectorSearchResult(entity, 1.0)
            return None
        except Exception as e:
            logger.error(f"Error retrieving vector {vector_id}: {e}")
            return None
    
    def get_documents_list(self) -> List[str]:
        """
        Get a list of all unique document names in the database.
        
        Returns:
            List of document names
        """
        all_entities = self.box.get_all()
        documents = set()
        
        for entity in all_entities:
            documents.add(entity.document)
        
        return sorted(list(documents))
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        all_entities = self.box.get_all()
        documents = set()
        total_content_length = 0
        
        for entity in all_entities:
            documents.add(entity.document)
            total_content_length += len(entity.content)
        
        # Get entity class name safely
        entity_class_name = getattr(self.entity_class, '__name__', str(self.entity_class))
        
        return {
            "total_vectors": len(all_entities),
            "unique_documents": len(documents),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "entity_class": entity_class_name,
            "average_content_length": total_content_length / len(all_entities) if all_entities else 0,
            "database_path": str(self.database_dir)
        }
    
    def close(self):
        """Close the database connection."""
        self.store.close()

def format_results(results: List[VectorSearchResult], max_content_length: int = 200) -> str:
    """Format search results for display."""
    if not results:
        return "No results found."
    
    output = []
    output.append(f"\nFound {len(results)} results:\n")
    output.append("=" * 80)
    
    for i, result in enumerate(results, 1):
        content_preview = result.content[:max_content_length]
        if len(result.content) > max_content_length:
            content_preview += "..."
        
        metadata_str = ", ".join([f"{k}: {v}" for k, v in result.metadata.items()][:3])
        
        output.append(f"\n{i}. Score: {result.score:.4f} | ID: {result.id}")
        output.append(f"   Document: {result.document}")
        output.append(f"   Name: {result.name}")
        output.append(f"   Content: {content_preview}")
        output.append(f"   Metadata: {metadata_str}")
        output.append("-" * 80)
    
    return "\n".join(output)

def save_results_json(results: List[VectorSearchResult], output_file: str):
    """Save search results to JSON file."""
    results_data = {
        "timestamp": int(time.time() * 1000),
        "total_results": len(results),
        "results": [result.to_dict() for result in results]
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Query ObjectBox vector database')
    parser.add_argument('--database-dir', type=str, required=True,
                       help='Directory containing the ObjectBox database')
    parser.add_argument('--query', type=str, help='Text query for similarity search')
    parser.add_argument('--text-search', type=str, help='Text to search for in content')
    parser.add_argument('--document', type=str, help='Get all chunks from specific document')
    parser.add_argument('--vector-id', type=int, help='Get specific vector by ID')
    parser.add_argument('--list-documents', action='store_true', help='List all documents in database')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results to return (default: 10)')
    parser.add_argument('--min-score', type=float, default=0.0, help='Minimum similarity score (default: 0.0)')
    parser.add_argument('--model', type=str, help='Override embedding model (for query encoding)')
    parser.add_argument('--output-json', type=str, help='Save results to JSON file')
    parser.add_argument('--search-field', type=str, default='content', 
                       choices=['content', 'name', 'document'],
                       help='Field to search in for text search (default: content)')
    
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--list-all', action='store_true', help='List all vectors in database (for debugging)')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    database_dir = Path(args.database_dir)
    if not database_dir.exists():
        logger.error(f"Database directory not found: {database_dir}")
        return
    
    # Initialize query engine
    try:
        query_engine = ObjectBoxQueryEngine(str(database_dir), args.model)
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        return
    
    try:
        results = []
        
        # Database statistics
        if args.stats:
            stats = query_engine.get_database_stats()
            print("\nDatabase Statistics:")
            print("=" * 50)
            for key, value in stats.items():
                print(f"{key}: {value}")
            return
        
        # List all vectors (debug)
        if args.list_all:
            all_entities = query_engine.box.get_all()
            print(f"\nAll vectors in database ({len(all_entities)} total):")
            print("=" * 80)
            for i, entity in enumerate(all_entities):
                print(f"\n{i+1}. ID: {entity.id}")
                print(f"   Name: {entity.name}")
                print(f"   Document: {entity.document}")
                print(f"   Content: {entity.content[:100]}...")
                #print(f"   Embedding length: {len(entity.embedding) if entity.embedding else 'None'}")
                try:
                    metadata = json.loads(entity.metadata) if entity.metadata else {}
                    print(f"   Metadata: {metadata}")
                except:
                    print(f"   Metadata: {entity.metadata}")
                print(f"   Timestamp: {entity.timestamp}")
                print("-" * 80)
                if i >= 10:  # Limit to first 10 for readability
                    print(f"... and {len(all_entities) - 11} more")
                    break
            return
        
        # Vector similarity search
        if args.query:
            logger.info(f"Performing vector similarity search for: '{args.query}'")
            results = query_engine.vector_similarity_search(
                args.query, 
                top_k=args.top_k, 
                min_score=args.min_score
            )
        
        # Text search
        elif args.text_search:
            logger.info(f"Performing text search for: '{args.text_search}'")
            results = query_engine.text_search(args.text_search, args.search_field)
            # Limit results for text search too
            results = results[:args.top_k]
        
        # Document retrieval
        elif args.document:
            logger.info(f"Retrieving document: '{args.document}'")
            results = query_engine.get_by_document(args.document)
        
        # Get by ID
        elif args.vector_id:
            logger.info(f"Retrieving vector ID: {args.vector_id}")
            result = query_engine.get_by_id(args.vector_id)
            if result:
                results = [result]
        
        else:
            print("Please specify a query type: --query, --text-search, --document, --vector-id, --list-documents, or --stats")
            return
        
        # Display results
        print(format_results(results))
        
        # Save to JSON if requested
        if args.output_json and results:
            save_results_json(results, args.output_json)
    
    finally:
        query_engine.close()

if __name__ == "__main__":
    main()