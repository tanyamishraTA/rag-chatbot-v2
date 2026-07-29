from app.pipelines.ingest_pipeline import IngestPipeline

pipeline = IngestPipeline(
    documents_path="documents",
)

pipeline.run()