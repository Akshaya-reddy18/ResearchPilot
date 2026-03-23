import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { getDocuments, createDocument } from "../../services/api";
import { deleteDocument } from "../../services/api";

export function WorkspaceDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [docs, setDocs] = useState<any[]>([]);
  const [title, setTitle] = useState("");

  const load = async () => {
    const data = await getDocuments(id!);
    setDocs(data);
  };

  const handleCreate = async () => {
    if (!title) return;
    await createDocument(id!, title);
    setTitle("");
    load();
  };

  useEffect(() => {
    load();
  }, [id]);

  return (
    <div className="flex-1 bg-gradient-to-br from-[#0a0e27] to-[#1a1f4d] text-white p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Workspace {id}</h1>
            <p className="text-sm text-gray-400">Manage documents in this workspace</p>
          </div>
          <button
            onClick={() => navigate('/workspaces')}
            className="px-3 py-2 rounded-lg bg-white/5 border border-indigo-900/40 text-sm text-gray-200 hover:bg-white/10"
          >
            Back to workspaces
          </button>
        </div>

        <div className="bg-white/5 border border-indigo-900/40 rounded-2xl p-4 md:p-6 space-y-4 backdrop-blur-xl">
          <div className="flex flex-col md:flex-row gap-2">
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="New document title"
              className="flex-1 px-3 py-2 rounded-lg bg-white/5 border border-indigo-900/40 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              onClick={handleCreate}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-sm font-medium shadow-lg shadow-indigo-500/30"
            >
              Create
            </button>
          </div>

          <div className="border-t border-white/10 pt-4 space-y-2">
            {docs.length === 0 ? (
              <p className="text-sm text-gray-400">
                No documents yet. Create your first document to start writing.
              </p>
            ) : (
              docs.map((doc) => (
                <div
                  key={doc.id}
                  onClick={() => navigate(`/doc/${doc.id}?workspace=${id}`)}
                  className="p-4 rounded-xl bg-white/5 border border-transparent hover:border-indigo-500/40 cursor-pointer transition-all flex items-center justify-between"
                >
                  <div>
                    <div className="text-sm font-medium">{doc.title || 'Untitled document'}</div>
                    <div className="text-xs text-gray-400">Click to open in editor</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={async (e) => {
                        e.stopPropagation();
                        if (!confirm('Delete this document? This cannot be undone.')) return;
                        try {
                          await deleteDocument(doc.id);
                          load();
                        } catch (err) {
                          alert('Failed to delete document');
                        }
                      }}
                      className="px-2 py-1 rounded text-xs bg-white/5 hover:bg-red-600/20"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}