import { useEffect, useRef, useState } from 'react'

export function useAutosave(editor: any, saveFn: (content: string) => Promise<void>, delay = 2000) {
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved'>('idle')
  const timeoutRef = useRef<any>(null)
  const lastContentRef = useRef<string>('')

  useEffect(() => {
    if (!editor) return

    const handler = () => {
      const newContent = editor.getHTML()
      if (newContent === lastContentRef.current) return
      setStatus('saving')
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(async () => {
        try {
          await saveFn(newContent)
          lastContentRef.current = newContent
          setStatus('saved')
          setTimeout(() => setStatus('idle'), 1200)
        } catch (err) {
          setStatus('idle')
        }
      }, delay)
    }

    editor.on('update', handler)
    return () => {
      editor.off('update', handler)
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [editor, saveFn, delay])

  return { autosaveState: status }
}
