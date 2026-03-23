import React, { useEffect, useState, useRef } from "react";
import { EditorContent, useEditor } from "@tiptap/react";
import StarterKit from "@tiptap/starter-kit";
import Image from "@tiptap/extension-image";
import {
  Bold,
  Italic,
  Heading1,
  Heading2,
  Heading3,
  ArrowLeft,
  Download,
  Image as ImageIcon,
  Sparkles,
} from "lucide-react";
import { chat, getDocumentById, updateDocument } from "../../services/api";
import { useNavigate } from "react-router-dom";

interface EditorProps {
  docId: number;
  workspaceId: string;
}

const API_BASE_URL =
  import.meta.env.VITE_API_URL ||
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

export default function Editor({ docId, workspaceId }: EditorProps) {
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("Idle");
  const [title, setTitle] = useState("");
  const [isDirty, setIsDirty] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const navigate = useNavigate();

  const editor = useEditor({
    extensions: [StarterKit, Image],
    content: "",
    editorProps: {
      attributes: {
        class:
          "prose prose-invert max-w-none focus:outline-none text-white min-h-[70vh]",
      },
    },
    onUpdate: ({ editor }) => {
      setStatus("Typing...");
      setIsDirty(true);

      const html = editor.getHTML();

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(
          JSON.stringify({ type: "edit", doc_id: docId, delta: { html } })
        );
      }
    },
  });

  // 🔹 Load + WebSocket
  useEffect(() => {
    if (!editor) return;

    const token = localStorage.getItem("token");
    const currentUserEmail = localStorage.getItem("user_email") || undefined;

    async function load() {
      try {
        const data = await getDocumentById(docId);
        setTitle(data.title || "");
        editor.commands.setContent(data.content || "");
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }

    load();

    const WS_BASE = API_BASE_URL.startsWith("https")
      ? API_BASE_URL.replace("https", "wss")
      : API_BASE_URL.replace("http", "ws");

    const ws = new WebSocket(
      `${WS_BASE}/ws/workspace/${workspaceId}?token=${token}`
    );

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);

        if (msg.type === "edit" && msg.doc_id === docId) {
          if (!editor.isFocused && msg.user?.email !== currentUserEmail) {
            editor.commands.setContent(msg.delta?.html || "");
          }
        }
      } catch (e) {
        console.error(e);
      }
    };

    wsRef.current = ws;

    return () => {
      ws.close();
      if (editor && !editor.isDestroyed) editor.destroy();
    };
  }, [docId, workspaceId, editor]);

  // 🔹 Autosave
  useEffect(() => {
    if (!editor) return;

    const interval = setInterval(async () => {
      if (!isDirty) return;

      try {
        setStatus("Saving...");
        const content = editor.getHTML();

        await updateDocument(docId, content, title || "Untitled document");

        setStatus("Saved");
        setIsDirty(false);
      } catch {
        setStatus("Error saving");
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [editor, isDirty, docId, title]);

  // 🔹 Navigation
  const handleBack = () => {
    navigate(`/workspace/${workspaceId}`);
  };

  // 🔹 Download FIXED
  const handleDownload = async (format: "pdf" | "docx") => {
    try {
      setStatus("Preparing download...");
      const token = localStorage.getItem("token");

      const res = await fetch(
        `${API_BASE_URL}/api/export/doc/${docId}/${format}`,
        {
          headers: {
            ...(token && { Authorization: `Bearer ${token}` }),
          },
        }
      );

      if (!res.ok) throw new Error("Download failed");

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = `${title || "document"}.${format}`;
      link.click();

      URL.revokeObjectURL(url);

      setStatus("Saved");
    } catch (e) {
      console.error(e);
      setStatus("Download failed");
      alert("Download failed. Try again.");
    }
  };

  // 🔹 AI Suggestions
  const requestAISuggestions = async () => {
    if (!editor) return;

    setStatus("Generating suggestions...");

    try {
      const res = await chat({
        query: editor.getText(),
        use_context: false,
      });

      const items = (res.analysis || "")
        .split(/\n\n|\n/)
        .map((s) => s.trim())
        .filter(Boolean);

      setSuggestions(items.slice(0, 6));
      setShowSuggestions(true);
      setStatus("Idle");
    } catch {
      setStatus("Error generating suggestions");
    }
  };

  // 🔹 Image Insert
  const insertImageFromURL = (url: string) => {
    if (!editor || !url || !url.trim()) return;
    let final = url.trim();
    // add https if scheme missing
    if (!/^https?:\/\//i.test(final) && !/^data:/i.test(final)) {
      final = `https://${final}`;
    }

    try {
      editor.chain().focus().setImage({ src: final }).run();
    } catch (e) {
      // fallback: insert raw HTML
      try {
        editor.chain().focus().insertContent(`<p><img src="${final}" alt=""/></p>`).run();
      } catch (err) {
        console.error('Failed to insert image', err);
      }
    }

    setIsDirty(true);
  };

  const insertImageFromFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => insertImageFromURL(reader.result as string);
    reader.readAsDataURL(file);
  };

  if (loading) {
    return (
      <div className="text-white p-6 animate-pulse">
        Loading editor...
      </div>
    );
  }

  return (
    <div className="flex-1 bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d] text-white p-6">
      <div className="max-w-5xl mx-auto space-y-4">

        {/* Header */}
        <div className="flex justify-between items-center">
          <button onClick={handleBack}>
            <ArrowLeft />
          </button>

          <input
            value={title}
            onChange={(e) => {
              setTitle(e.target.value);
              setIsDirty(true);
            }}
            className="text-2xl bg-transparent outline-none"
            placeholder="Untitled"
          />

          <span>{status}</span>
        </div>

        {/* Toolbar */}
        <div className="flex gap-2">
          <button onClick={() => editor.chain().focus().toggleBold().run()}>
            <Bold />
          </button>
          <button onClick={() => editor.chain().focus().toggleItalic().run()}>
            <Italic />
          </button>
          <button onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}>
            <Heading1 />
          </button>
          <button onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}>
            <Heading2 />
          </button>
          <button onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}>
            <Heading3 />
          </button>

          <button onClick={() => handleDownload("pdf")}>
            <Download /> PDF
          </button>

          <button onClick={requestAISuggestions}>
            <Sparkles />
          </button>

          <button
            onClick={() => {
              const url = prompt("Image URL");
              if (url) insertImageFromURL(url);
            }}
          >
            <ImageIcon />
          </button>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) insertImageFromFile(f);
            }}
            className="block"
          />
        </div>

        {/* Editor */}
        <div className="bg-white/5 p-4 rounded-xl min-h-[70vh]">
          <EditorContent editor={editor} />
        </div>

        {/* AI Suggestions */}
        {showSuggestions && (
          <div className="fixed bottom-6 right-6 w-80 bg-black p-4 rounded">
            {suggestions.map((s, i) => (
              <div key={i} className="mb-2">
                {s}
                <button
                  onClick={() => {
                    editor.chain().focus().insertContent(`<p>${s}</p>`).run();
                    setShowSuggestions(false);
                  }}
                >
                  Insert
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}