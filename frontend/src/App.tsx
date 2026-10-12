import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type Citation = {
  source: string
  section: string
  chunk_id: string
}

type ChatResponse = {
  answer: string
  category: string
  need_retrieval: boolean
  confidence: number
  citations: Citation[]
}

const API_URL = 'http://127.0.0.1:8000/chat'

function App() {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState<ChatResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const trimmedQuestion = question.trim()
    if (!trimmedQuestion || loading) return

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmedQuestion, top_k: 3 }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `请求失败（${response.status}）`)
      }

      setResult((await response.json()) as ChatResponse)
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '请求失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="page-shell">
      <header className="page-header">
        <p className="eyebrow">ENTERPRISE KNOWLEDGE BASE</p>
        <h1>企业知识库问答</h1>
        <p className="subtitle">基于制度原文检索，返回可追溯的答案。</p>
      </header>

      <section className="workspace" aria-label="问答区域">
        <form className="question-form" onSubmit={handleSubmit}>
          <label htmlFor="question">请输入你的问题</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="例如：年假多少天？"
            rows={4}
            disabled={loading}
          />
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? '正在查询...' : '提交问题'}
          </button>
        </form>

        <section className="answer-panel" aria-live="polite">
          <div className="panel-heading">
            <h2>回答</h2>
            {result && <span className="confidence">置信度 {result.confidence.toFixed(2)}</span>}
          </div>

          {loading && <p className="muted">正在检索知识库并生成答案...</p>}
          {error && <p className="error">{error}</p>}
          {!loading && !error && !result && <p className="muted">提交问题后，答案会显示在这里。</p>}
          {result && <p className="answer-text">{result.answer}</p>}

          {result && result.citations.length > 0 && (
            <div className="citations">
              <h3>来源</h3>
              <ul>
                {result.citations.map((citation) => (
                  <li key={citation.chunk_id}>
                    <span>{citation.source}</span>
                    <span>{citation.section}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </section>
      </section>
    </main>
  )
}

export default App
