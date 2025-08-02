from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Array, Float32, String, Int64
from feast.value_type import ValueType
from datetime import timedelta

# Define document entity
document = Entity(
    name="document_id",
    value_type=ValueType.STRING,
    description="Unique identifier for document chunks"
)

# Define document embeddings data source (for batch loading)
document_embeddings_source = FileSource(
    name="document_embeddings_source",
    path="data/document_embeddings.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Define push source for real-time ingestion
document_embeddings_push_source = PushSource(
    name="document_embeddings_push_source",
    batch_source=document_embeddings_source,
)

# Define document embeddings feature view with push source
document_embeddings = FeatureView(
    name="document_embeddings",
    entities=[document],
    ttl=timedelta(days=365),
    schema=[
        Field(name="vector", dtype=Array(Float32), vector_index=True),  # Official naming
        Field(name="item_id", dtype=Int64),  # Official pattern
        Field(name="chunk_text", dtype=String),
        Field(name="document_title", dtype=String), 
        Field(name="chunk_index", dtype=Int64),
        Field(name="file_path", dtype=String),
        Field(name="chunk_length", dtype=Int64),
    ],
    online=True,
    source=document_embeddings_push_source,  # Use push source as primary
    tags={"team": "rag", "version": "v3"},
)