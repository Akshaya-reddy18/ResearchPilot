import React, { useRef } from 'react'
import { Editor } from '@tiptap/react'
import { Bold, Italic, Image as ImageIcon, FileText, DownloadCloud, Loader2, Sparkles } from 'lucide-react'

type Props = {
  editor: Editor | null
  onUploadImage: (file: File) => Promise<void>
  onExportPDF: () => Promise<void>
  onExportDOCX: () => Promise<void>
  exporting: boolean
  autosaveState: 'saved' | 'saving' | 'idle'
  onAISuggest?: () => Promise<void>
}

export const EditorToolbar: React.FC<Props> = ({ editor, onUploadImage, onExportPDF, onExportDOCX, exporting, autosaveState, onAISuggest }) => {
  const fileRef = useRef<HTMLInputElement | null>(null)

  const handleImageClick = () => fileRef.current?.click()

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    try {
      const file = e.target?.files?.[0]
      if (file) await onUploadImage(file)
    } catch (err) {
      console.error('file input error', err)
    } finally {
      try {
        e.currentTarget.value = ''
      } catch {}
    }
  }

  return (
    <div className="sticky top-4 z-20 bg-white/80 dark:bg-slate-900/80 backdrop-blur-md rounded-md py-2 px-3 shadow-sm border border-slate-100 dark:border-slate-800 transition">
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-2">
          <button
            onClick={() => { try { editor?.chain().focus().toggleBold().run() } catch (err) { console.warn('bold failed', err) } }}
            className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-slate-700 dark:text-slate-200"
            aria-label="Bold"
            disabled={!editor}
          >
            <Bold className="w-4 h-4" />
          </button>

          <button
            onClick={() => { try { editor?.chain().focus().toggleItalic().run() } catch (err) { console.warn('italic failed', err) } }}
            className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-slate-700 dark:text-slate-200"
            aria-label="Italic"
            disabled={!editor}
          >
            <Italic className="w-4 h-4" />
          </button>

          <div className="border-l h-6 mx-2" />

          <div className="inline-flex items-center gap-1">
            <button
              onClick={() => { try { editor?.chain().focus().toggleHeading({ level: 1 }).run() } catch (err) { console.warn('h1 failed', err) } }}
              className="px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-sm text-slate-700 dark:text-slate-200"
              disabled={!editor}
            >
              H1
            </button>
            <button
              onClick={() => { try { editor?.chain().focus().toggleHeading({ level: 2 }).run() } catch (err) { console.warn('h2 failed', err) } }}
              className="px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-sm text-slate-700 dark:text-slate-200"
              disabled={!editor}
            >
              H2
            </button>
            <button
              onClick={() => { try { editor?.chain().focus().toggleHeading({ level: 3 }).run() } catch (err) { console.warn('h3 failed', err) } }}
              className="px-2 py-1 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-sm text-slate-700 dark:text-slate-200"
              disabled={!editor}
            >
              H3
            </button>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={onFileChange} />
          <button onClick={handleImageClick} className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-slate-700 dark:text-slate-200" aria-label="Upload image" disabled={!onUploadImage}>
            <ImageIcon className="w-4 h-4" />
          </button>

          <button onClick={() => onAISuggest && onAISuggest()} className="p-2 rounded hover:bg-slate-100 dark:hover:bg-slate-800 transition text-slate-700 dark:text-slate-200" aria-label="AI Suggest">
            <Sparkles className="w-4 h-4" />
          </button>

          <button onClick={onExportPDF} className="flex items-center gap-2 px-3 py-1 rounded bg-sky-600 text-white hover:bg-sky-700 transition shadow-sm" disabled={exporting}>
            {exporting ? <Loader2 className="w-4 h-4 animate-spin" /> : <FileText className="w-4 h-4" />}
            <span className="text-sm">Export PDF</span>
          </button>

          <button onClick={onExportDOCX} className="flex items-center gap-2 px-3 py-1 rounded border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition">
            <DownloadCloud className="w-4 h-4" />
            <span className="text-sm">Export DOCX</span>
          </button>
        </div>

        <div className="ml-4 text-sm text-slate-500 dark:text-slate-400">{autosaveState === 'saving' ? 'Saving...' : autosaveState === 'saved' ? 'Saved' : ''}</div>
      </div>
    </div>
  )
}

export default EditorToolbar
