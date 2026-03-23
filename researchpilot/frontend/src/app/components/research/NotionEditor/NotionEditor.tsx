import React, { useCallback, useEffect, useState } from 'react'
import { useEditor, EditorContent } from '@tiptap/react'
import StarterKit from '@tiptap/starter-kit'
import Placeholder from '@tiptap/extension-placeholder'
import Image from '@tiptap/extension-image'
import Heading from '@tiptap/extension-heading'
import EditorToolbar from './EditorToolbar'
import { useAutosave } from './hooks/useAutosave'
import { updateDocument, exportDocumentPDF, exportDocumentDOCX, aiSuggest } from '../../../services/api'
import showToast from './utils/toast'
import { DownloadCloud } from 'lucide-react'

type Props = {
  docId: string | number
  initialContent?: string
  title?: string
}

export const NotionEditor: React.FC<Props> = ({ docId, initialContent = '', title = 'Untitled' }) => {
  const [exporting, setExporting] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const editor = useEditor({
    extensions: [
      StarterKit.configure({ heading: false }),
      Heading.configure({ levels: [1, 2, 3] }),
      Image,
      Placeholder.configure({ placeholder: 'Start writing your research...' }),
    ],
    content: initialContent,
    editorProps: {
      attributes: { class: 'prose max-w-none focus:outline-none' },
    },
  })

  const saveFn = useCallback(async (html: string) => {
    await updateDocument(Number(docId), html)
  }, [docId])

  const { autosaveState } = useAutosave(editor, saveFn, 1800)

  useEffect(() => {
    // load Inter font if not present
    const id = 'inter-font'
    if (!document.getElementById(id)) {
      const link = document.createElement('link')
      link.id = id
      link.rel = 'stylesheet'
      link.href = 'https://rsms.me/inter/inter.css'
      document.head.appendChild(link)
    }
  }, [])

  // Ensure editor is ready before applying initial content
  useEffect(() => {
    let mounted = true
    async function applyContent() {
      try {
        if (!editor) return
        // Wait until editor is initialized
        if (!editor.isDestroyed) {
          // Only set content if it differs
          const current = editor.getHTML()
          if ((initialContent || '').trim() && current !== initialContent) {
            try {
              editor.commands.setContent(initialContent)
            } catch (err) {
              console.warn('Failed to set editor content', err)
            }
          }
        }
      } catch (err) {
        console.error('applyContent error', err)
      } finally {
        if (mounted) setLoading(false)
      }
    }

    applyContent()

    return () => {
      mounted = false
    }
  }, [editor, initialContent])

  const handleUploadImage = async (file: File) => {
    try {
      const reader = new FileReader()
      reader.onload = () => {
        const src = String(reader.result)
        try {
          if (editor) editor.chain().focus().setImage({ src }).run()
        } catch (err) {
          console.warn('Insert image failed', err)
        }
        showToast('Image inserted', 'success')
      }
      reader.readAsDataURL(file)
    } catch (e) {
      showToast('Image upload failed', 'error')
    }
  }

  const doExportPDF = async () => {
    setExporting(true)
    try {
      const blob = await exportDocumentPDF(Number(docId))
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${title || 'document'}.pdf`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      showToast('PDF exported', 'success')
    } catch (err) {
      console.error(err)
      showToast('Export failed', 'error')
    } finally {
      setExporting(false)
    }
  }

  const doExportDOCX = async () => {
    setExporting(true)
    try {
      const blob = await exportDocumentDOCX(Number(docId))
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${title || 'document'}.docx`
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
      showToast('DOCX exported', 'success')
    } catch (err) {
      console.error(err)
      showToast('Export DOCX failed', 'error')
    } finally {
      setExporting(false)
    }
  }

  const [aiLoading, setAiLoading] = useState(false)
  const [aiSuggestion, setAiSuggestion] = useState<string | null>(null)

  const handleAISuggest = async () => {
    if (!editor) return
    setAiLoading(true)
    try {
      const content = editor.getText()
      const res: any = await aiSuggest(content, 'Provide concise writing suggestions and improvements:')
      const suggestion = res?.analysis || res?.analysis_text || res?.analysis || JSON.stringify(res)
      setAiSuggestion(String(suggestion))
    } catch (e) {
      console.error(e)
      showToast('AI suggestion failed', 'error')
    } finally {
      setAiLoading(false)
    }
  }

  const insertAISuggestion = () => {
    if (!editor || !aiSuggestion) return
    editor.chain().focus().insertContent(`<p>${aiSuggestion}</p>`).run()
    setAiSuggestion(null)
    showToast('Suggestion inserted', 'success')
  }

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 text-slate-800 dark:text-slate-100 font-sans">
      <header className="w-full bg-gradient-to-r from-sky-700 via-indigo-700 to-violet-700 text-white py-6 shadow-md">
        <div className="max-w-6xl mx-auto px-4 flex items-center gap-4">
          <div className="text-lg font-semibold">ResearchPilot</div>
          <div className="ml-auto hidden sm:block">
            <div className="text-sm opacity-90">{title}</div>
          </div>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8 flex gap-6">

        <div className="flex-1">
          <div className="mx-auto max-w-3xl">
            <EditorToolbar
              editor={editor}
              onUploadImage={handleUploadImage}
              onExportPDF={doExportPDF}
              onExportDOCX={doExportDOCX}
              exporting={exporting}
              onAISuggest={handleAISuggest}
              autosaveState={autosaveState === 'saving' ? 'saving' : autosaveState === 'saved' ? 'saved' : 'idle'}
            />

            <div className="relative mt-4">
              {loading && (
                <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60 dark:bg-slate-900/60 rounded-lg">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-sky-600" />
                </div>
              )}

              <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm p-6 prose prose-slate dark:prose-invert max-h-[70vh] overflow-y-auto leading-7">
                {error ? (
                  <div className="text-sm text-rose-500">{error}</div>
                ) : (
                  <EditorContent editor={editor} />
                )}
              </div>

              {/* AI Suggestion panel */}
              {aiSuggestion !== null && (
                <div className="mt-4 bg-slate-50 dark:bg-slate-900 rounded-md p-4 border border-slate-100 dark:border-slate-800">
                  <div className="flex items-start gap-4">
                    <div className="prose prose-slate dark:prose-invert flex-1 text-sm">{aiSuggestion}</div>
                    <div className="flex flex-col gap-2">
                      <button onClick={insertAISuggestion} className="px-3 py-1 bg-sky-600 text-white rounded">Insert</button>
                      <button onClick={() => setAiSuggestion(null)} className="px-3 py-1 border rounded">Close</button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

export default NotionEditor
