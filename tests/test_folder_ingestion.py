import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import embeddings as embeddings_module
from core.ingestion import Chunk, batch_ingest_folder


class BatchIngestFolderTests(unittest.TestCase):
    def test_batch_ingest_folder_skips_already_ingested_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            data_dir.mkdir()

            pdf_one = data_dir / "paper_one.pdf"
            pdf_two = data_dir / "paper_two.pdf"
            pdf_one.write_bytes(b"%PDF-1.4\nfake pdf")
            pdf_two.write_bytes(b"%PDF-1.4\nsecond fake pdf")

            def fake_process_pdf(pdf_path, chunk_size=400, overlap=50):
                return [
                    Chunk(
                        id=f"{Path(pdf_path).stem}_chunk",
                        text="sample chunk",
                        source=Path(pdf_path).name,
                        page=1,
                        chunk_index=0,
                    )
                ]

            class DummyVectorDB:
                def __init__(self):
                    self.added = []

                def add_chunks(self, embedded_chunks):
                    self.added.extend(embedded_chunks)

            def fake_embed_chunks(chunks):
                return [
                    type("EmbeddedChunk", (), {"chunk": chunk, "embedding": [0.1, 0.2]})()
                    for chunk in chunks
                ]

            with patch("core.ingestion.process_pdf", side_effect=fake_process_pdf), patch.object(
                embeddings_module, "embed_chunks", side_effect=fake_embed_chunks
            ):
                first_result = batch_ingest_folder(str(data_dir), vector_db=DummyVectorDB())
                second_result = batch_ingest_folder(str(data_dir), vector_db=DummyVectorDB())

            self.assertEqual(first_result["ingested_files"], ["paper_one.pdf", "paper_two.pdf"])
            self.assertEqual(first_result["skipped_files"], [])
            self.assertEqual(second_result["ingested_files"], [])
            self.assertEqual(second_result["skipped_files"], ["paper_one.pdf", "paper_two.pdf"])
            self.assertTrue((data_dir / ".ingested_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
