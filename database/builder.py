#!/usr/bin/env python3
# DECOMPOSE(#5): RAG/vector-DB helpers move to src/mobiletransformers/rag
# (ingestion/embeddings/vector_store) in Tier 2 (#25/#26); Android ObjectBox stays in the Android module. ~28 KB.
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
import shutil
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


class MobileTransformersObjectBoxProcessor:
    def __init__(self, database_dir: str, embedding_dim: Optional[int] = None, model_name: str = "all-MiniLM-L6-v2", no_embed: bool = False):
        self.database_dir = Path(database_dir)
        self.model_name = model_name
        self.no_embed = no_embed
        
        # Create database directory
        self.database_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.no_embed:
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
        else:
            # When no embedding, use provided dimension or default to 384
            logger.info("No embedding mode enabled - storing empty vectors")
            self.embeddings = None
            self.embedding_dim = embedding_dim if embedding_dim is not None else 384
            logger.info(f"Using embedding dimension for empty vectors: {self.embedding_dim}")
        
        # Validate dimension is supported
        if self.embedding_dim not in [64, 128, 256, 384, 512, 768, 1024, 1536]:
            logger.error(f"Specified embedding dimension {self.embedding_dim} is not supported.")
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
        
        entity_class, _, entity_id = entity_configs[self.embedding_dim]
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
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'},
                    recursive=True,
                    show_progress=True
            )
            md_docs = md_loader.load()
            logger.info(f"Loaded {len(md_docs)} markdown files (preserving formatting)")
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
                           splitter_type: str = "recursive", markdown_headers: Optional[List[str]] = None) -> Any:
        """Create a text splitter using LangChain."""
        if splitter_type == "recursive":
            separators = ["\n\n", "\n", " ", ""]
            
            # If markdown headers are specified, add them as primary separators
            if markdown_headers:
                logger.info(f"Using custom markdown headers as separators: {markdown_headers}")
                # Convert markdown headers to actual header patterns
                header_separators = []
                for header in markdown_headers:
                    if header.startswith('#'):
                        header_separators.append(f"\n{header} ")
                    else:
                        # Assume it's a header level like "##" or "###"
                        header_separators.append(f"\n{header} ")
                separators = header_separators + separators
            
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=separators
            )
        elif splitter_type == "token":
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "markdown":
            # For markdown splitter, we can specify headers to split on
            if markdown_headers:
                logger.info(f"Using MarkdownHeaderTextSplitter with headers: {markdown_headers}")
                # Convert to the format expected by MarkdownHeaderTextSplitter
                headers_to_split_on = []
                for header in markdown_headers:
                    if header.startswith('#'):
                        level = len(header.split()[0])  # Count the # symbols
                        headers_to_split_on.append((header.split()[0], f"Header_{level}"))
                    else:
                        # Assume it's just the # symbols
                        level = len(header)
                        headers_to_split_on.append((header, f"Header_{level}"))
                
                from langchain.text_splitter import MarkdownHeaderTextSplitter
                return MarkdownHeaderTextSplitter(
                    headers_to_split_on=headers_to_split_on,
                    return_each_line=True,
                    strip_headers=False
                )
            else:
                return MarkdownTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
        elif splitter_type == "document":
            logger.info("Using document-based splitter - each document becomes one chunk")
            return None  # We'll handle this case specially in process_and_store
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
    
    def process_and_store(self, documents: List[Document], text_splitter: Any, splitter_type,
                         batch_size: int = 100) -> int:
        """Process documents and store in ObjectBox using custom entities."""
        
        # Handle document-based splitter (no actual splitting)
        if text_splitter is None:  # document splitter case
            logger.info("Using document-based chunking - storing whole documents")
            chunks = documents  # Each document is a chunk
        elif splitter_type == "markdown":
            # Handle MarkdownHeaderTextSplitter which uses split_text instead of split_documents
            logger.info("Splitting documents into chunks using MarkdownHeaderTextSplitter...")
            chunks = []
            for doc in documents:
                # MarkdownHeaderTextSplitter.split_text returns Documents directly
                doc_chunks = text_splitter.split_text(doc.page_content)
                # Add original document metadata to each chunk
                for chunk in doc_chunks:
                    chunk.metadata.update(doc.metadata)
                chunks.extend(doc_chunks)
        else:
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
                
                # Generate embeddings for the batch or create empty vectors
                if not self.no_embed and self.embeddings:
                    logger.info(f"Generating embeddings for {len(texts)} chunks...")
                    embeddings = self.embeddings.embed_documents(texts)
                else:
                    logger.info(f"Creating empty vectors for {len(texts)} chunks (no-embed mode)")
                    # Create empty vectors with the correct dimension
                    embeddings = [[0.0] * self.embedding_dim for _ in range(len(texts))]
                
                # Create entity instances
                entities = []
                for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                    # Extract source file name
                    source = chunk.metadata.get('source', f'document_{i + j}')
                    document_name = Path(source).name if source else f'document_{i + j}'

                    #print(chunk.page_content)
                    #print("------------------------")
                    
                    # Create chunk metadata
                    chunk_metadata = {
                        'chunk_id': i + j,
                        'chunk_index': j
                    }
                    
                    # Add any existing metadata from the document
                    if hasattr(chunk, 'metadata') and chunk.metadata:
                        chunk_metadata.update(chunk.metadata)
                    
                    # Create entity
                    entity_name = document_name if text_splitter is None else f"{document_name}_chunk_{j}"
                    entity = self.create_vector_entity(
                        name=entity_name,
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
            "database_path": str(self.database_dir),
            "no_embed_mode": self.no_embed
        }
    
    def close(self):
        """Close the database."""
        self.store.close()

def process_documents_with_custom_entities(input_dir: Path, output_dir: Path, embedding_dim: Optional[int],
                                         model_name: str = "all-MiniLM-L6-v2", 
                                         chunk_size: int = 512, chunk_overlap: int = 50,
                                         splitter_type: str = "recursive", no_embed: bool = False,
                                         markdown_headers: Optional[List[str]] = None):
    """Main processing function using custom ObjectBox entities."""
    
    # Initialize processor (embedding_dim can be None for auto-inference)
    processor = MobileTransformersObjectBoxProcessor(str(output_dir), embedding_dim, model_name, no_embed)
    
    try:
        # Load documents using LangChain loaders
        logger.info("Loading documents...")
        documents = processor.load_documents(input_dir)
        
        if not documents:
            logger.error("No documents found to process")
            return
        
        # Create text splitter
        if splitter_type == "document":
            logger.info("Using document-based splitter - no chunking will be performed")
            text_splitter = None
        else:
            logger.info(f"Creating {splitter_type} text splitter (chunk_size={chunk_size}, overlap={chunk_overlap})")
            if markdown_headers:
                logger.info(f"Using custom markdown headers: {markdown_headers}")
            text_splitter = processor.create_text_splitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                splitter_type=splitter_type,
                markdown_headers=markdown_headers
            )
        
        # Process and store documents
        logger.info("Processing and storing documents...")
        total_stored = processor.process_and_store(documents, text_splitter, splitter_type)
        
        # Get final statistics
        stats = processor.get_database_stats()
        
        logger.info("Database creation completed!")
        logger.info(f"Documents processed: {len(documents)}")
        logger.info(f"Total vectors stored: {stats['total_vectors']}")
        logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
        logger.info(f"Model used: {stats['model_name']}")
        logger.info(f"No-embed mode: {stats['no_embed_mode']}")
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
            "distance_type": "cosine",
            "no_embed_mode": no_embed,
            "markdown_headers": markdown_headers
        }
        
        config_file = output_dir / "processing_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Processing configuration saved to {config_file}")
        
    finally:
        processor.close()

def validate_and_prepare_schema(script_dir: Path):
    """
    Validate that default.json exists and copy it to objectbox-model/default.json
    in the script directory before database creation.
    """
    # Check for default.json in the same directory as the script
    kotlin_schema_path = script_dir / "default.json"
    
    if not kotlin_schema_path.exists():
        logger.error("="*60)
        logger.error("SCHEMA ERROR: Missing Kotlin ObjectBox schema!")
        logger.error("="*60)
        logger.error(f"Required file not found: {kotlin_schema_path}")
        logger.error("")
        logger.error("SOLUTION:")
        logger.error("1. Copy your Kotlin app's 'objectbox-models/default.json' file")
        logger.error(f"2. Place it in the same directory as this script: {script_dir}")
        logger.error("3. Rename it to 'default.json'")
        logger.error("")
        logger.error("This file is required to ensure Python entities match")
        logger.error("your Kotlin ObjectBox schema (IDs, UIDs, indexes).")
        logger.error("="*60)
        raise FileNotFoundError(f"Kotlin ObjectBox schema not found: {kotlin_schema_path}")
    
    # Validate the JSON file
    try:
        with open(kotlin_schema_path, 'r') as f:
            schema_data = json.load(f)
        
        # Basic validation
        if 'entities' not in schema_data:
            raise ValueError("Invalid ObjectBox schema: missing 'entities' field")
        
        if not schema_data['entities']:
            raise ValueError("Invalid ObjectBox schema: no entities found")
        
        # Check for VectorEntity classes
        vector_entities = [e for e in schema_data['entities'] if e['name'].startswith('VectorEntity')]
        if not vector_entities:
            logger.warning("No VectorEntity classes found in schema. Expected VectorEntity64, VectorEntity384, etc.")
        else:
            entity_names = [e['name'] for e in vector_entities]
            logger.info(f"Found VectorEntity classes: {', '.join(entity_names)}")
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema file: {e}")
        raise ValueError(f"Kotlin schema file is not valid JSON: {kotlin_schema_path}")
    except Exception as e:
        logger.error(f"Error validating schema: {e}")
        raise
    
    # Create objectbox-model directory in script directory
    
    # Copy schema to the expected location in script directory
    target_schema_path = script_dir / "objectbox-model.json"
    shutil.copy2(kotlin_schema_path, target_schema_path)
    
    logger.info(f"✅ Kotlin schema validated and copied to: {target_schema_path}")
    logger.info(f"📋 Schema contains {len(schema_data['entities'])} entities")
    
    return target_schema_path, schema_data

def main():
    parser = argparse.ArgumentParser(description='Create ObjectBox vector database with custom entities using LangChain')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Directory containing documents to process')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for the ObjectBox database')
    parser.add_argument('--embedding-dim', type=int, default=None,
                       choices=[64, 128, 256, 384, 512, 768, 1024, 1536],
                       help='Embedding dimension (default: infer from model, or 384 if --no-embed)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                       help='HuggingFace model name (default: sentence-transformers/all-MiniLM-L6-v2)')
    parser.add_argument('--chunk-size', type=int, default=128,
                       help='Text chunk size (default: 128)')
    parser.add_argument('--chunk-overlap', type=int, default=32,
                       help='Chunk overlap size (default: 32)')
    parser.add_argument('--splitter-type', type=str, default='recursive',
                       choices=['recursive', 'token', 'markdown', 'document'],
                       help='Text splitter type (default: recursive)')
    parser.add_argument('--no-embed', action='store_true',
                       help='Skip embedding generation and store empty vectors')
    parser.add_argument('--markdown-headers', type=str, nargs='*',
                       help='Markdown headers to split on (e.g., "##" "###" or "## Section" "### Subsection")')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    script_dir = Path(__file__).parent.resolve()
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Validate markdown headers format if provided
    markdown_headers = None
    if args.markdown_headers:
        markdown_headers = args.markdown_headers
        logger.info(f"Will use markdown headers for splitting: {markdown_headers}")
    
    # Validate and prepare ObjectBox schema BEFORE anything else
    try:
        schema_path, schema_data = validate_and_prepare_schema(script_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Schema validation failed: {e}")
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
    logger.info(f"No-embed mode: {args.no_embed}")
    logger.info(f"Markdown headers: {markdown_headers}")
    
    # Call the main processing function
    process_documents_with_custom_entities(
        input_dir=input_dir,
        output_dir=output_dir,
        embedding_dim=args.embedding_dim,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        splitter_type=args.splitter_type,
        no_embed=args.no_embed,
        markdown_headers=markdown_headers
    )

    logger.info("Copying the generated object-box model json to output dir")
    # Copy the schema file to output directory
    try:
        shutil.copy2(schema_path, output_dir)
        logger.info(f"Successfully copied schema file to {output_dir / schema_path.name}")
    except Exception as e:
        logger.error(f"Failed to copy schema file: {e}")

if __name__ == "__main__":
    main()