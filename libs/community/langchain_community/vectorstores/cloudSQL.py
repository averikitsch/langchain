import uuid

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from google.cloud.sql.connector import Connector
import google.auth
from google.auth.transport.requests import Request
from google.auth.compute_engine import _metadata

from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import JSON, UUID

from pgvector.sqlalchemy import Vector

def _get_IAM_user(credentials):
    """Get user/service account name"""

    # Checks for service account
    if hasattr(credentials, "service_account_email"):
        return credentials.service_account_email[:-20]
    else:
        # Defaults to getting the user account
        command = ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr}"
        
class CloudSQLEngine:
    """Creating a connection to the CloudSQL instance
    To use, you need the following packages installed:
        cloud-sql-python-connector[asyncpg]
    """
    
    connector = Connector()
    
    @staticmethod
    def from_instance(
        region: str, 
        instance: str, 
        database: str, 
        project_id: str=None
    ) -> sqlalchemy.engine.Engine:
     """Create sqlalchemy connection to the postgres database in the CloudSQL instance.

        Args:
            region (str): CloudSQL instance region.
            instance (str): CloudSQL instance name.
            database (str): CloudSQL instance database name.
            project_id (str): GCP project ID. Defaults to None

        Returns:
            Sqlalchemy engine containing the connection pool.
        """

        if project_id is None:
            credentials, project_id = google.auth.default()

        IAM_USER = _get_IAM_user(credentials)
        
        conn = connector.connect(
            INSTANCE_URI = f"{project_id}:{region}:{instance}",
            'asyncpg',
            user = IAM_USER,
            db = database,
            enable_iam_auth = True
        )
        
        pool = sqlalchemy.create_engine("postgresql+asyncpg://",creator=conn)
        
        return pool
        
class CloudSQLVectorStore(VectorStore):
    """Google Cloud SQL vector store.

    To use, you need the following packages installed:
        pgvector-python
        sqlalchemy
        cloud-sql-python-connector[asyncpg]
    """
    
    def __init__(
        self, 
        engine: CloudSQLEngine, 
        embedding_service: Embeddings, 
        table_name: str, 
        content_column: str='content', 
        embedding_column: str='embedding', 
        metadata_columns: Optional[str, List[str]]='metadata', 
        ignore_metadata_columns: Optional[str, List[str]]=None, 
        overwrite_existing: bool=False, 
        index_query_options=None, 
        distance_strategy: str="L2"
    ) -> None:
        """Constructor for CloudSQLVectorStore.

        Args:
            engine (sqlalchemy.engine.Engine): Sqlalchmey engine with pool connection to the postgres database. Required.
            embedding_service (Embeddings): Text embedding model to use.
            table_name (str): Name of the existing table or the table to be created.
            content_column (str): Column that represent a Document’s page_content. Defaults to content
            embedding_column (str): Column for embedding vectors. 
                              The embedding is generated from the document value. Defaults to embedding
            metadata_columns (List[str]): Column(s) that represent a document's metadata. Defaults to metadata
            ignore_metadata_columns (List[str]): Column(s) to ignore in pre-existing tables for a document’s metadata. 
                                     Can not be used with metadata_columns. Defaults to None
            overwrite_existing (bool): Boolean for truncating table before inserting data. Defaults to False
            index_query_options : QueryOptions class with vector search parameters. Defaults to None
            distance_strategy (str):
                Determines the strategy employed for calculating
                the distance between vectors in the embedding space.
                Defaults to EUCLIDEAN_DISTANCE(L2).
                Available options are:
                - COSINE: Measures the similarity between two vectors of an inner
                    product space.
                - EUCLIDEAN_DISTANCE: Computes the Euclidean distance between
                    two vectors. This metric considers the geometric distance in
                    the vector space, and might be more suitable for embeddings
                    that rely on spatial relationships. This is the default behavior.
            """

        self.engine = engine
        self.embedding_service = embedding_service
        self.table_name = table_name
        self.content_column = content_column
        self.embedding_column = embedding_column
        self.metadata_columns = metadata_columns
        self.ignore_metadata_columns = ignore_metadata_columns
        self.overwrite_existing = overwrite_existing
        self.index_query_options = index_query_options
        self.distance_strategy = distance_strategy
        self.__post_init__()
        
    def __post_init__(self) -> None:
        """Initialize table and validate existing tables"""
        self.create_vector_extension()

        if self.overwrite_existing:
            drop_stmt = sqlalchemy.text(f"DROP TABLE IF EXISTS {self.table_name}")
            with self.engine.connect() as connection:
                connection.execute(drop_stmt)
                connection.commit()
            self.create_table()

        # If both metadata and ignore_metadata are given, throw an error
        if self.metadata_columns is not None and self.ignore_metadata_columns is not None:
            raise ValueError("Both metadata_columns and ignore_metadata_columns have been provided.")

        if self.engine.dialect.has_table(self.engine.connect(), self.table_name):
            inspector = inspect(self.engine)
            column_names = [column['name'] for column in inspector.get_columns(self.table_name)]
            if self.embedding_column not in column_names:
                raise ValueError(f"Column {self.embedding_column} does not exist")
            if self.content_column not in column_names:
                raise ValueError(f"Column {self.content_column} does not exist")
            if 'metadata' not in column_names:
                raise ValueError(f"Column 'metadata' does not exist")
            if 'uuid' not in column_names:
                raise ValueError(f"Column 'uuid' does not exist")

        self.create_default_table(metadata)

    def embeddings(
        self
    ) -> Embeddings:
        return self.embedding_service

    def create_vector_extension(
        self
    ) -> None:
        """Creates the vector extsion to the specified database."""
        query = sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector")
        with self.engine.connect() as connection:
            connection.execute(query)
            connection.commit()

    def create_default_table(
        self
    ) -> None:
        """Creates the default table."""
        if self.engine.dialect.has_table(self.engine.connect(), self.table_name) and self.overwrite_existing == False:
            raise ValueError("Table already exists.")
        else:
            query = f'''
               CREATE TABLE IF NOT EXISTS {self.table_name} (
                   uuid UUID PRIMARY KEY,
                   content VARCHAR(255) NOT NULL,
                   embedding vector NOT NULL,
                   metadata JSON NOT NULL
               );
               '''
            with self.engine.connect() as connection:
                result = connection.execute(sqlalchemy.text(query))
                connection.commit()

    
    def init_vector_store(
        self,  
        table_name: str, 
        vector_size: int, 
        content_column: str='content', 
        embedding_column: str='embedding', 
        metadata_columns: Optional[str, List[str]]='metadata',
        index: HNSWIndex,
        overwrite_existing: bool=False, 
        store_metadata: bool=True
    ) -> None:

        """Creating a non-default vectorstore table"""
        query = f'''
        CREATE TABLE IF NOT EXISTS {table_name} (
            uuid UUID PRIMARY KEY,
            {content_column} VARCHAR(255) NOT NULL,
            {embedding_column} vector({vector_size}) NOT NULL,
            {metadata_columns} JSON NOT NULL
        );
        '''
        with self.engine.connect() as connection:
            result = connection.execute(text(query))
            connection.commit()
    
    @classmethod
    def from_documents(
        cls, 
        docs: List[Document], 
        engine: sqlalchemy.engine.Engine, 
        table_name: str,
        embedding_service: Embeddings,
        ids: Optional[List[str]] = None
    ) -> CloudSQLVectorStore:
         """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        embeddings = embedding_service.embed_documents(list(texts))

        table = cls(
            engine=engine,
            embedding_service=embedding_service,
            table_name=table_name,
        )

        table.add_embeddings(
            texts=texts, engine=engine, embeddings=embeddings, metadatas=metadatas, ids=ids, table_name=table_name)
        
        return table
    
    @classmethod
    def from_texts(
        cls, 
        texts: List[str],
        metadatas: List[Dict]=None,
        embedding_service: Embeddings,
        engine: sqlalchemy.engine.Engine, 
        table_name: str,
        ids: Optional[List[str]]=None
    ) -> CloudSQLVectorStore:
        """ Return VectorStore initialized from texts and embeddings."""
        if not metadatas:
            metadatas = [{} for _ in texts]

        documents = []
        for text, meta in zip(texts, metadatas):
            docs = Document(page_content = text, metadata=meta)
            documents.append(docs)

        return cls.from_documents(
            engine=engine,
            documents=documents,
            embedding_service=embedding_service,
            table_name=table_name,
            ids=ids)

    def add_embeddings(
        self,
        engine: CloudSQLEngine,
        texts: List[str],
        embeddings: Embeddings,
        metadatas: List[float]=None,
        ids: Optional[List[str]]=None,
        table_name: str
    ) -> List[str]:

        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]

        with engine.connect() as connection:
            for id, content, embedding, meta in zip(ids, texts, embeddings, metadatas):
                data_to_add = {"ids":id, 'content':content, 'embedding':embedding, 'metadata':meta}
                stmt = f"INSERT INTO {table_name}(ids, content, embedding, metadata) VALUES (:ids, :content, :embedding, :metadata)"
                connection.execute(sqlalchemy.text(stmt), data_to_add)
                connection.commit()

        return ids

    def add_documents(
        self, 
        documents: List[Document], 
        ids: Optional[List[str]]=None
        ) -> List[str]:
         """Run more documents through the embeddings and add to the vectorstore.

        Args:
            documents (List[Document]): Iterable of Documents to add to the vectorstore.
            ids (List[str]): List of id strings. Defaults to None

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        embeddings = self.embedding_service.embed_documents(list(texts))

        return self.add_embeddings(
            texts=texts, 
            embeddings=embeddings, 
            metadatas=metadatas, 
            ids=ids, 
            engine=self.Engine, 
            table_name=self.table_name)


    def add_texts(
        self, 
        texts: Iterable[str], 
        metadatas: Optional[List[dict]]=None, 
        ids: Optional[List[str]]=None
        ) -> List[str]:
         """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (str): Iterable of strings to add to the vectorstore.
            metadatas (List[dict]): Optional list of metadatas associated with the texts. Defaults to None.
            ids (List[str]): List of id strings. Defaults to None

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        if not metadata:
            metadata = [{} for _ in texts]

        documents = []
        for text, meta in zip(texts, metadata):
            docs = Document(page_content = text, metadata=meta)
            documents.append(docs)

        return add_documents(
            documents=documents,
            ids=ids)