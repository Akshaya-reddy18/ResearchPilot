import { useParams, useLocation } from "react-router-dom";
import React, { useEffect, useState } from "react";
import NotionEditor from "./NotionEditor/NotionEditor";
import { getDocumentById } from "../../services/api";

export function EditorPage() {
  const { id } = useParams();
  const location = useLocation();
  const params = new URLSearchParams(location.search);
  const workspaceId = params.get("workspace") || "default";
  const [initialContent, setInitialContent] = useState("");
  const [title, setTitle] = useState<string | undefined>(undefined);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    let mounted = true;
    (async () => {
      try {
        setLoading(true);
        const data = await getDocumentById(Number(id));
        if (!mounted) return;
        setInitialContent(data?.content || "");
        setTitle(data?.title || undefined);
      } catch (e) {
        console.error('Failed to load document', e);
        setError('Failed to load document. Showing empty editor.');
        setInitialContent('');
        setTitle(undefined);
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [id]);

  return (
    <div className="flex-1 p-6">
      {loading && <div className="mb-4 text-sm text-slate-500">Loading document…</div>}
      {error && <div className="mb-4 text-sm text-rose-500">{error}</div>}
      <NotionEditor docId={Number(id)} initialContent={initialContent} title={title} />
    </div>
  );
}