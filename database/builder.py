#!/usr/bin/env python3
"""
ObjectBox Vector Database Precompute Script with LangChain

Creates an ObjectBox database with precomputed embeddings using LangChain tools for document processing
and text chunking. The generated database can be transferred to mobile devices.

Usage:
    python objectbox_precompute_langchain.py --input-dir ./documents --output-dir ./database --embedding-dim 384
"""


import json
import argparse
from pathlib import Path
import time
from typing import List, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from objectbox import Store
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
    from langchain_objectbox.vectorstores import ObjectBox as LangChainObjectBox
    from langchain_community.document_loaders import (
        TextLoader, 
        DirectoryLoader,
        JSONLoader,
        UnstructuredMarkdownLoader
    )
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
        MarkdownTextSplitter
    )
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"LangChain dependencies not installed: {e}")
    logger.error("Install with: pip install langchain-community langchain-objectbox")
    exit(1)

class ORTMobileObjectBoxProcessor:
    def __init__(self, database_dir: str, embedding_dim: Optional[int] = None, model_name: str = "all-MiniLM-L6-v2"):
        self.database_dir = Path(database_dir)
        self.model_name = model_name
        
        # Create database directory
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings first to get the actual dimension
        logger.info(f"Initializing HuggingFace embeddings with model: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Infer embedding dimension from model if not provided
        if embedding_dim is None:
            # Test embedding to get dimension
            test_embedding = self.embeddings.embed_query("test")
            inferred_dim = len(test_embedding)
            logger.info(f"Inferred embedding dimension from model: {inferred_dim}")
            self.embedding_dim = inferred_dim
        else:
            self.embedding_dim = embedding_dim
            # Verify the provided dimension matches the model
            test_embedding = self.embeddings.embed_query("test")
            actual_dim = len(test_embedding)
            if actual_dim != embedding_dim:
                logger.warning(f"Provided embedding dimension ({embedding_dim}) doesn't match model dimension ({actual_dim}). Using model dimension: {actual_dim}")
                self.embedding_dim = actual_dim
        
        # Validate dimension is supported
        if self.embedding_dim not in [64, 128, 256, 384, 512, 768, 1024, 1536]:
            logger.error(f"Model produces {self.embedding_dim}D embeddings, which is not supported.")
            logger.error("Supported dimensions: 64, 128, 256, 384, 512, 768, 1024, 1536")
            raise ValueError(f"Unsupported embedding dimension: {self.embedding_dim}")
        
        logger.info(f"Using embedding dimension: {self.embedding_dim}")
        
        # Setup ObjectBox model and store
        self.entity_class = self._get_entity_class(self.embedding_dim)
        self.model = Model()
        self._setup_model()
        
        # Create ObjectBox store and box
        self.store = Store(model=self.model, directory=str(self.database_dir))
        self.box = self.store.box(self.entity_class)
        
        logger.info(f"ObjectBox database initialized at {self.database_dir}")
    
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
        
        if embedding_dim not in entity_map:
            raise ValueError(f"Unsupported embedding dimension: {embedding_dim}. "
                           f"Supported dimensions: {list(entity_map.keys())}")
        
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
    
    def load_documents(self, input_dir: Path) -> List[Document]:
        """Load documents using LangChain loaders."""
        documents = []
        
        # Load text files
        try:
            text_loader = DirectoryLoader(
                str(input_dir),
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                recursive=True,
                show_progress=True
            )
            text_docs = text_loader.load()
            documents.extend(text_docs)
            logger.info(f"Loaded {len(text_docs)} text files")
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")
        
        # Load markdown files
        try:
            md_loader = DirectoryLoader(
                str(input_dir),
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader,
                recursive=True,
                show_progress=True
            )
            md_docs = md_loader.load()
            documents.extend(md_docs)
            logger.info(f"Loaded {len(md_docs)} markdown files")
        except Exception as e:
            logger.warning(f"Error loading markdown files: {e}")
        
        # Load JSON files
        try:
            for json_file in input_dir.rglob("*.json"):
                try:
                    json_loader = JSONLoader(
                        file_path=str(json_file),
                        jq_schema='.', 
                        text_content=False
                    )
                    json_docs = json_loader.load()
                    documents.extend(json_docs)
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
            logger.info(f"Loaded JSON files")
        except Exception as e:
            logger.warning(f"Error loading JSON files: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def create_text_splitter(self, chunk_size: int = 512, chunk_overlap: int = 50, 
                           splitter_type: str = "recursive") -> Any:
        """Create a text splitter using LangChain."""
        if splitter_type == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "markdown":
            return MarkdownTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unknown splitter type: {splitter_type}")
    
    def create_vector_entity(self, name: str, content: str, document: str, 
                           embedding: List[float], metadata: Dict[str, Any]) -> Any:
        """Create a vector entity instance."""
        entity = self.entity_class()
        entity.name = name
        entity.content = content
        entity.document = document
        entity.embedding = embedding
        entity.metadata = json.dumps(metadata)
        entity.timestamp = int(time.time() * 1000)  # Current timestamp in milliseconds
        return entity
    
    def insert_vector_batch(self, entities: List[Any]) -> List[int]:
        """Insert multiple vector entities in a batch."""
        try:
            ids = []
            for entity in entities:
                entity_id = self.box.put(entity)
                ids.append(entity_id)
            return ids
        except Exception as e:
            logger.error(f"Error inserting batch: {e}")
            return []
    
    def process_and_store(self, documents: List[Document], text_splitter: Any, 
                         batch_size: int = 100) -> int:
        """Process documents and store in ObjectBox using custom entities."""
        logger.info("Splitting documents into chunks...")
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        # Process chunks in batches
        total_stored = 0
        current_timestamp = int(time.time() * 1000)
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            try:
                # Prepare texts for embedding
                texts = [chunk.page_content for chunk in batch_chunks]
                
                # Generate embeddings for the batch
                logger.info(f"Generating embeddings for {len(texts)} chunks...")
                embeddings = self.embeddings.embed_documents(texts)
                
                # Create entity instances
                entities = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    # Extract source file name
                    source = chunk.metadata.get('source', f'document_{i + j}')
                    document_name = Path(source).name if source else f'document_{i + j}'
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        'chunk_id': i + j,
                        'chunk_index': j,
                    }
                    
                    # Add any existing metadata from the document
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        chunk_metadata.update(chunk.metadata)
                    
                    # Create entity
                    entity = self.create_vector_entity(
                        name=f"{document_name}_chunk_{j}",
                        content=chunk.page_content,
                        document=document_name,
                        embedding=embedding,
                        metadata=chunk_metadata
                    )
                    entities.append(entity)
                
                # Insert batch
                logger.info(f"Inserting {len(entities)} entities into database...")
                ids = self.insert_vector_batch(entities)
                
                if ids:
                    total_stored += len(ids)
                    logger.info(f"Successfully stored {len(ids)} chunks. Total: {total_stored}")
                else:
                    logger.error(f"Failed to store batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        return total_stored
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_vectors": self.box.count(),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "database_path": str(self.database_dir)
        }
    
    def close(self):
        """Close the database."""
        self.store.close()

def process_documents_with_custom_entities(input_dir: Path, output_dir: Path, embedding_dim: Optional[int],
                                         model_name: str = "all-MiniLM-L6-v2", 
                                         chunk_size: int = 512, chunk_overlap: int = 50,
                                         splitter_type: str = "recursive"):
    """Main processing function using custom ObjectBox entities."""
    
    # Initialize processor (embedding_dim can be None for auto-inference)
    processor = ORTMobileObjectBoxProcessor(str(output_dir), embedding_dim, model_name)
    
    try:
        # Load documents using LangChain loaders
        logger.info("Loading documents...")
        documents = processor.load_documents(input_dir)
        
        if not documents:
            logger.error("No documents found to process")
            return
        
        # Create text splitter
        logger.info(f"Creating {splitter_type} text splitter (chunk_size={chunk_size}, overlap={chunk_overlap})")
        text_splitter = processor.create_text_splitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            splitter_type=splitter_type
        )
        
        # Process and store documents
        logger.info("Processing and storing documents...")
        total_stored = processor.process_and_store(documents, text_splitter)
        
        # Get final statistics
        stats = processor.get_database_stats()
        
        logger.info("Database creation completed!")
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Total vectors stored: {stats['total_vectors']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"Model used: {stats['model_name']}")
        logger.info(f"Database location: {stats['database_path']}")
        
        # Save database info
        info_file = output_dir / "database_info.json"
        with open(info_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Database info saved to {info_file}")
        
        # Save processing configuration for reference
        config = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "splitter_type": splitter_type,
            "model_name": model_name,
            "embedding_dimension": stats['embedding_dimension'],
            "total_documents": len(documents),
            "total_vectors": stats['total_vectors'],
            "supported_file_types": [".txt", ".md", ".json"],
            "distance_type": "cosine"  # Default for your Android implementation
        }
        
        config_file = output_dir / "processing_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Processing configuration saved to {config_file}")
        
    finally:
        processor.close()

def main():
    parser = argparse.ArgumentParser(description='Create ObjectBox vector database with custom entities using LangChain')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing documents to process')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for the ObjectBox database')
    parser.add_argument('--embedding-dim', type=int, default=None,
                       choices=[64, 128, 256, 384, 512, 768, 1024, 1536],
                       help='Embedding dimension (default: infer from model)')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='HuggingFace model name (default: all-MiniLM-L6-v2)')
    parser.add_argument('--chunk-size', type=int, default=512,
                       help='Text chunk size (default: 512)')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Chunk overlap size (default: 50)')
    parser.add_argument('--splitter-type', type=str, default='recursive',
                       choices=['recursive', 'token', 'markdown'],
                       help='Text splitter type (default: recursive)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ObjectBox vector database creation with custom entities...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Embedding dimension: {'auto-infer from model' if args.embedding_dim is None else args.embedding_dim}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    logger.info(f"Splitter type: {args.splitter_type}")
    
    process_documents_with_custom_entities(
        input_dir, output_dir, args.embedding_dim, 
        args.model, args.chunk_size, args.chunk_overlap, args.splitter_type
    )
    
    logger.info("\n" + "="*50)
    logger.info("TRANSFER INSTRUCTIONS:")
    logger.info("="*50)
    logger.info(f"1. Copy the entire directory '{output_dir}' to your Android device")
    logger.info("2. Place it in your app's cache directory or assets")
    logger.info("3. The main database file is: {}/data.mdb".format(output_dir))
    logger.info("4. Use the database_info.json file to configure your ORTRagConfig")
    logger.info("5. Processing configuration is saved in processing_config.json")
    logger.info("6. Vector similarity will use COSINE distance (default for Android)")
    logger.info("="*50)

if __name__ == "__main__":
    main()