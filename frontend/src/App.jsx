import { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

function App() {
  const [file, setFile] = useState(null);
  const [uploadSummary, setUploadSummary] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState(null);
  const [citations, setCitations] = useState([]);
  const [status, setStatus] = useState('Ready');
  const [error, setError] = useState(null);

  const handleUpload = async (event) => {
    event.preventDefault();
    setError(null);
    setStatus('Uploading...');
    setUploadSummary(null);

    if (!file) {
      setError('Please select a PDF file first.');
      setStatus('Ready');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || payload.error || 'Upload failed');
      }

      const data = await response.json();
      setUploadSummary(data.summary ?? data);
      setStatus('Upload completed.');
    } catch (err) {
      setError(err.message);
      setStatus('Upload failed.');
    }
  };

  const handleQuery = async (event) => {
    event.preventDefault();
    setError(null);
    setStatus('Querying...');
    setAnswer(null);
    setCitations([]);

    if (!question.trim()) {
      setError('Please enter a question.');
      setStatus('Ready');
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question, top_k: 5 }),
      });

      if (!response.ok) {
        const payload = await response.json();
        throw new Error(payload.detail || payload.error || 'Query failed');
      }

      const data = await response.json();
      setAnswer(data.answer);
      setCitations(data.citations || []);
      setStatus('Query completed.');
    } catch (err) {
      setError(err.message);
      setStatus('Query failed.');
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Research Paper RAG UI</h1>
        <p>Upload a PDF and ask questions against the uploaded documents.</p>
      </header>

      <section className="panel">
        <h2>Upload PDF</h2>
        <form onSubmit={handleUpload}>
          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button type="submit">Upload &amp; Ingest</button>
        </form>

        {uploadSummary && (
          <div className="summary-card">
            <strong>Upload summary:</strong>
            <pre>{JSON.stringify(uploadSummary, null, 2)}</pre>
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Ask a question</h2>
        <form onSubmit={handleQuery}>
          <textarea
            rows="4"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question here"
          />
          <button type="submit">Ask</button>
        </form>

        {answer && (
          <div className="summary-card">
            <strong>Answer:</strong>
            <p>{answer}</p>
            {citations.length > 0 && (
              <div>
                <strong>Citations:</strong>
                <ul>
                  {citations.map((citation) => (
                    <li key={`${citation.source}-${citation.page}-${citation.marker}`}>
                      [{citation.marker}] {citation.source}, page {citation.page}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>

      <footer>
        <div>Status: {status}</div>
        {error && <div className="error-message">{error}</div>}
      </footer>
    </div>
  );
}

export default App;
